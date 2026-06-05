# Novel Contribution / Real Research

> **The problem:** The current pipeline uses someone else's KG (OptimusKG), someone else's algorithm (AMIE), and someone else's LLM. The engineering is solid, but the novelty is thin. The ideas below are what would make this *yours*.

## 1. Two-Stage Neural→Symbolic Link Prediction (Main Contribution)

Train a KGE model (RotatE/ComplEx via PyKEEN) for fast, fuzzy link prediction (high recall), then filter those predictions through AMIE's mined rules for hard-logic validation (high precision).

- **Why it's novel:** Prior work either trains KGEs alone OR mines rules alone. A few papers jointly embed rules during training (KALE, RUGE), but nobody has done a two-stage "generate then filter" pipeline on a heterogeneous biomedical KG like OptimusKG, and nobody has measured how much symbolic filtering improves precision over the neural model alone.
- **Your contribution:** The architecture + the empirical finding that symbolic filtering improves precision on biomedical link prediction without destroying recall.
- **Concrete deliverable:** Train RotatE via PyKEEN → predict top-N links → filter with AMIE rules → compare precision/recall/MRR against RotatE-alone and AMIE-alone. Three bars on a chart. That's a paper.

## 2. Rule-Guided Negative Sampling for KGE Training (Enhancement to #1)

Every KGE model uses **random negative sampling** — randomly corrupting triples by swapping head/tail with a random entity. Most random negatives are trivially easy, leading to weak training signal.

- **The idea:** Use AMIE's mined rules to generate **hard negatives** — triples that *almost* satisfy a discovered rule but violate exactly one condition. E.g., if AMIE discovered `(?a CONTRAINDICATION ?b) ∧ (?b ASSOCIATED_WITH ?c) ⇒ (?a ASSOCIATED_WITH ?c)`, generate a negative by finding `(a,b)` where CONTRAINDICATION holds, `(b,c)` where ASSOCIATED_WITH holds, but `(a,c)` is known to be FALSE.
- **Why it's novel:** Rule-guided negative sampling for KGE training on biomedical KGs doesn't exist. The closest work is adversarial negative sampling (KBGAN), but that uses a GAN, not symbolic rules.
- **Pairs with #1:** This improves the KGE training step inside the same pipeline, giving a two-contribution paper.

## 3. LLM-as-Biological-Plausibility-Filter for Mined Rules

AMIE mines rules purely from graph structure — it has no idea whether a rule makes biological sense. Some rules are structural artifacts with no pharmacological meaning.

- **The idea:** After AMIE mines N rules, feed each rule to an LLM: *"This logical rule was mined from a biomedical KG. Assess its biological plausibility and explain why it would or wouldn't hold in pharmacology."* Use the LLM's assessment to filter out structurally valid but biologically meaningless rules before prediction.
- **Why it's novel:** Nobody has used LLMs to post-filter symbolic rule miners for biological validity. It's a genuine neuro-symbolic loop: symbolic mining → neural validation → symbolic prediction.
- **Caveat:** Harder to evaluate rigorously — need a way to measure "plausibility" beyond LLM opinion.

### Priority Ranking

| Idea | Novelty | Feasibility | Paper Potential |
|---|---|---|---|
| **#1 KGE→AMIE two-stage** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Strong. Clear experiments. |
| **#2 Rule-guided negatives** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Highest novelty, needs careful impl. |
| **#3 LLM rule plausibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Novel but harder to evaluate. |

---

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

## 4. Relationship Ontology Cleanup (Biological Embedding Clustering)
- **Goal:** Resolve relationship redundancy (e.g., `IS_A` vs `PARENT`) without losing biological nuance (e.g., keeping `INHIBITOR` and `ANTAGONIST` distinct).
- **Approach:** Use a biological embedding model (e.g., BioBERT, PubMedBERT) to embed the relationship types.
- Perform semantic clustering on the embeddings to automatically merge true synonyms while preserving biologically distinct mechanisms.

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