# Phoenix : Probabilistic Hybrid Open-source Enhanced NLP with Integrated eXpertise
Project PHOENIX: Probabilistic Hybrid Open-source Enhanced NLP with Integrated eXpertise

## Finetuning GPT-Neo
Our project involves using a open source model and finetune it with the objective of teaching it how to reason using bayesian inference. Our model of choice is GPT-Neo, similar to GPT-2 architecture and open source were one of the main reason as to why we went ahead with this model. 
 The benchmarking dataset that we choose to use was the Bayesian Linguistic Inference Dataset [BlinD](https://github.com/HLR/BLInD). Though this dataset comes from a non peer reviewed, it is a well documented dataset that contains the following : 
 - A foundational Bayesian Network for each instance.
- A textual description detailing the structure of the Bayesian Network.
- Probabilistic queries posed in natural language.
- Precise answers corresponding to these queries.

It contains about 1000 queries that containing the following subsets: 
- Context 
- Query 
- Answer 
- Graph 
- Depth
- Reasoning 
- Solution 
- query_type
- query_state
- calculation_time

While all of these are important, we only need to focus on the top 7 concepts, we use some data preprocessing methods to restructure the dataset to a more cleaner JSON format 
- Input = Context + Query 
- Reasoning = Graph+ Depth 
- Answer = Answers 

This format allows me to break down these part into easier digestable part for the model to understand to be finetuned on. We will mention on how we used this format for finetuning. 
