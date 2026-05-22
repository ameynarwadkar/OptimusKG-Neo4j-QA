# Future Enhancements & TODOs

## 1. Extract 2-Hop Diabetes Subgraph from OptimusKG
- Run `scripts/get_diabetes_2hop.py` on the university cluster to extract a massive 2-hop neighborhood for "diabetes".
- **Goal:** Provide AMIE with cross-disease, distant biological data to discover highly novel, unexpected genetic targets that are completely unlinked in a restricted 1-hop view.

## 2. Integrate AMIE Rules into the LLM / QA System
- Inject the parsed AMIE rules (`outputs/discovered_rules.csv` or `amie_discovered_rules.md`) into the RAG context.
- **Goal:** Enable the QA system to reason over missing links mathematically. (e.g., When asked "Why is Drug X related to Alzheimer's?", the LLM can cite Rule 6: "Because it is contraindicated, implying a shared genetic target pathway").

## 3. Scale Up on the University Cluster
- Modify `scripts/export_for_amie.py` to export the entire OptimusKG, not just the Alzheimer's subset.
- Increase `MAX_RULE_LENGTH` to 5 in `scripts/run_amie.py`.
- **Goal:** Leverage the cluster's 128 cores and 64GB+ RAM to discover universal pharmacological rules across all diseases and drugs globally in the database.

## 4. Relationship Ontology Cleanup (Redundancy Resolution)
- **Goal:** Merge redundant relationship types before running rule mining to improve AMIE's precision and recall.
- Map ontological relationships (`IS_A` $\rightarrow$ `PARENT`).
- Group pharmacological down-regulators (`INHIBITOR`, `ANTAGONIST`, `BLOCKER`, etc. $\rightarrow$ `INHIBITOR`).
- Group pharmacological up-regulators (`AGONIST`, `ACTIVATOR`, etc. $\rightarrow$ `ACTIVATOR`).

## 5. Implement Neuro-Symbolic Hybrid Pipeline
- Address AMIE's limitations (symbolic logic rigidity) by integrating Knowledge Graph Embeddings (KGEs) or Graph Neural Networks (GNNs).
- **Goal:** Build a robust discovery engine that uses Neural Networks (e.g., PyKEEN/DGL) for fast, fuzzy link prediction (high recall), and filters those predictions through AMIE for hard-logic validation and explainability (high precision).


The Ultimate "Hybrid" Pipeline (Neuro-Symbolic AI)
If you rely only on neural networks (KGEs/GNNs), you lose AMIE's best feature: Explainability. Neural networks are black boxes. AMIE tells you exactly why it made a prediction.

Here is how you combine them into a State-of-the-Art pipeline:

Step 1: The Fast Recall Layer (The Neural Network)
You train a Knowledge Graph Embedding model (e.g., using the Python library PyKEEN or DGL). You ask it to predict the top 10,000 most likely missing links in your graph.

Pros: It's lightning-fast and catches all the fuzzy, implicit connections AMIE misses.
Cons: It occasionally hallucinates and cannot explain why it made the prediction.
Step 2: The Logic Filter (AMIE)
You take those 10,000 predictions from the neural network and feed them into AMIE. You ask AMIE: "Can you find a logical, explainable rule that supports this prediction?"

If AMIE finds a rule, the prediction is upgraded to High Confidence & Explainable.
If AMIE cannot find a rule, the prediction is marked as Speculative / Unverified.
Step 3: The LLM Validator (The "Common Sense" Layer)
You take the final list of high-confidence predictions and feed them into a Large Language Model (like GPT-4 or Gemini) via an API. You prompt it: "My graph algorithm predicts that [Method X] will solve [Task Y] because of [AMIE Rule]. Does this make scientific sense based on your training data?"

Why this is incredibly powerful:
High Recall: The neural network finds the hidden, fuzzy connections.
High Precision & Explainability: AMIE acts as a BS-detector, filtering out neural network hallucinations using hard logic.
Real-world Grounding: The LLM acts as the final human-like peer review.
If you coded this pipeline in Python (combining Neo4j + AMIE + PyKEEN + LLM API), you would have built an enterprise-grade discovery engine that rivals what top tech companies use today.