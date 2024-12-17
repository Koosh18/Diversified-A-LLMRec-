import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM
import numpy as np

class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        num_reasoning_chains=5,  # Number of reasoning chains to generate
        mmr_lambda=0.5,  # Trade-off parameter for MMR
    ):
        super().__init__()
        self.device = device
        
        if llm_model == 'opt':
            self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, load_in_8bit=True, device_map=self.device)
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
        else:
            raise Exception(f'{llm_model} is not supported')
            
        # Add special tokens
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]','[CandidateEmb]']})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        # Freeze model parameters
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        self.max_output_txt_len = max_output_txt_len
        self.num_reasoning_chains = num_reasoning_chains
        self.mmr_lambda = mmr_lambda

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        history_token_id = self.llm_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        candidate_token_id = self.llm_tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        
        for inx in range(len(llm_tokens["input_ids"])):
            idx_tensor=(llm_tokens["input_ids"][inx]==history_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        
            idx_tensor=(llm_tokens["input_ids"][inx]==candidate_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        return llm_tokens, inputs_embeds
    
    def generate_reasoning_chains(self, input_context, num_chains=None):
        """
        Generate multiple reasoning chains for a given input context
        
        Args:
            input_context (str): Initial context for reasoning
            num_chains (int, optional): Number of reasoning chains to generate
        
        Returns:
            list: Diversified reasoning chains
        """
        if num_chains is None:
            num_chains = self.num_reasoning_chains
        
        reasoning_chains = []
        
        # Generate initial diverse prompts to seed different reasoning paths
        seed_prompts = self._create_diverse_seed_prompts(input_context)
        
        for prompt in seed_prompts[:num_chains]:
            current_context = input_context
            chain = []
            
            # Generate multiple reasoning steps
            for step in range(3):  # Typically 3 reasoning steps
                # Generate next reasoning step
                next_step = self._generate_reasoning_step(current_context)
                chain.append(next_step)
                
                # Update context dynamically
                current_context += " " + next_step
            
            reasoning_chains.append(" ".join(chain))
        
        return reasoning_chains
    
    def _create_diverse_seed_prompts(self, input_context):
        """
        Create diverse seed prompts to initiate different reasoning paths
        
        Args:
            input_context (str): Initial input context
        
        Returns:
            list: Diverse seed prompts
        """
        seed_prompts = [
            f"Let's break down {input_context} step by step.",
            f"What are the key insights we can derive from {input_context}?",
            f"Analyze {input_context} from multiple perspectives.",
            f"Explore the underlying patterns in {input_context}.",
            f"Critically examine {input_context} with a systematic approach."
        ]
        return seed_prompts
    
    def _generate_reasoning_step(self, context):
        """
        Generate a single reasoning step given the current context
        
        Args:
            context (str): Current reasoning context
        
        Returns:
            str: Next reasoning step
        """
        # Tokenize the context
        inputs = self.llm_tokenizer(context, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Generate the next reasoning step
        outputs = self.llm_model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        
        # Decode the generated text
        reasoning_step = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return reasoning_step
    
    def apply_mmr_selection(self, reasoning_chains):
        """
        Apply Maximal Marginal Relevance (MMR) to select diverse reasoning chains
        
        Args:
            reasoning_chains (list): List of reasoning chains
        
        Returns:
            list: Selected diverse reasoning chains
        """
        # Compute embedding representations of reasoning chains
        chain_embeddings = self._compute_chain_embeddings(reasoning_chains)
        
        # Apply MMR selection
        selected_indices = self._mmr_selection(chain_embeddings, reasoning_chains)
        
        return [reasoning_chains[i] for i in selected_indices]
    
    def _compute_chain_embeddings(self, reasoning_chains):
        """
        Compute embeddings for reasoning chains
        
        Args:
            reasoning_chains (list): List of reasoning chains
        
        Returns:
            numpy.ndarray: Embeddings for reasoning chains
        """
        # Use the LLM to generate embeddings
        embeddings = []
        for chain in reasoning_chains:
            inputs = self.llm_tokenizer(chain, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.llm_model(**inputs, output_hidden_states=True)
                # Use the last hidden state's mean as the embedding
                chain_embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                embeddings.append(chain_embedding.flatten())
        
        return np.array(embeddings)
    
    def _compute_query_embedding(self, query):
        """
        Compute embedding for the original query
        
        Args:
            query (str): Original input query
        
        Returns:
            numpy.ndarray: Query embedding
        """
        # Tokenize the query
        inputs = self.llm_tokenizer(query, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.llm_model(**inputs, output_hidden_states=True)
            # Use the last hidden state's mean as the embedding
            query_embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        
        return query_embedding.flatten()

    def _mmr_selection(self, embeddings, chains, query=None):
        """
        Perform MMR-based selection of reasoning chains
        
        Args:
            embeddings (numpy.ndarray): Embeddings of reasoning chains
            chains (list): Original reasoning chains
            query (str, optional): Original input query for relevance computation
        
        Returns:
            list: Indices of selected reasoning chains
        """
        num_select = min(len(chains), self.num_reasoning_chains)
        selected_indices = []
        
        # Compute query embedding if not provided
        if query is not None:
            query_embedding = self._compute_query_embedding(query)
        else:
            # If no query, use the mean of chain embeddings as a proxy
            query_embedding = embeddings.mean(axis=0)
        
        # Compute cosine similarity for relevance
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Select the first chain (most relevant to query)
        relevance_scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        
        # Select subsequent chains
        while len(selected_indices) < num_select:
            candidate_scores = []
            
            for i in range(len(embeddings)):
                if i in selected_indices:
                    continue
                
                # Compute relevance to query
                relevance_score = cosine_similarity(query_embedding, embeddings[i])
                
                # Compute diversity (dissimilarity to already selected chains)
                diversity_scores = []
                for j in selected_indices:
                    # Compute 1 - cosine similarity to measure dissimilarity
                    diversity_scores.append(1 - cosine_similarity(embeddings[i], embeddings[j]))
                
                # Use minimum diversity score (most dissimilar to existing selections)
                diversity_score = min(diversity_scores) if diversity_scores else 1
                
                # Compute MMR score
                mmr_score = (1 - self.mmr_lambda) * relevance_score + self.mmr_lambda * diversity_score
                candidate_scores.append((i, mmr_score))
            
            # Select the chain with the highest MMR score
            next_idx = max(candidate_scores, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)
        
        return selected_indices

    def inference(self, input_context):
        """
        Perform inference using MMR-based diversification
        
        Args:
            input_context (str): Input context for inference
        
        Returns:
            list: Diversified reasoning chains
        """
        # Generate reasoning chains
        reasoning_chains = self.generate_reasoning_chains(input_context)
        
        # Compute embeddings for reasoning chains
        chain_embeddings = self._compute_chain_embeddings(reasoning_chains)
        
        # Apply MMR selection with the original query context
        selected_indices = self._mmr_selection(chain_embeddings, reasoning_chains, query=input_context)
        
        return [reasoning_chains[i] for i in selected_indices]
    
    def forward(self, log_emb, samples):
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
            
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100))

        targets = torch.cat([empty_targets, targets], dim=1)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, samples['interact'], samples['candidate'])
        attention_mask = llm_tokens['attention_mask']
        
        log_emb = log_emb.unsqueeze(1)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        
        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss
    
    def inference(self, input_context):
        """
        Perform inference using MMR-based diversification
        
        Args:
            input_context (str): Input context for inference
        
        Returns:
            list: Diversified reasoning chains
        """
        # Generate reasoning chains
        reasoning_chains = self.generate_reasoning_chains(input_context)
        
        # Apply MMR selection
        diversified_chains = self.apply_mmr_selection(reasoning_chains)
        
        return diversified_chains