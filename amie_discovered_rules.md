# AMIE Rule Mining Results: Alzheimer's Subgraph

This document outlines the predictive rules discovered by AMIE (Inductive Logic Programming) on the OptimusKG Alzheimer's subgraph. AMIE successfully analyzed 60,000+ facts to find biological patterns capable of inferring new relationships.

---

## 1. Phenotype-Driven Genetic Inference
**Rule:**
`IF (?f ASSOCIATED_WITH ?b) AND (?a PHENOTYPE_PRESENT ?f) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 99.9%
* **Support:** 2,641

**Meaning & Conclusion:**
If a specific Gene (`?b`) is associated with a Phenotype (`?f`), and a Disease (`?a`) presents that exact Phenotype, then the Disease is associated with that Gene.
**Conclusion:** AMIE independently learned the concept of "phenotypic overlap." Diseases sharing physical manifestations (phenotypes) with a gene's known effects are highly likely to be driven by that same gene.

**Example:**
If the *APOE* gene (`?b`) is associated with the phenotype *Memory Impairment* (`?f`), and *Alzheimer's Disease* (`?a`) presents with *Memory Impairment* (`?f`), then *Alzheimer's Disease* (`?a`) is associated with the *APOE* gene (`?b`).

---

## 2. Network Pharmacology via Contraindication (The "Harm" Principle)
**Rule:** 
`IF (?j ASSOCIATED_WITH ?b) AND (?e ASSOCIATED_WITH ?j) AND (?e CONTRAINDICATION ?a) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 99.8%
* **Support:** 956

**Meaning & Conclusion:**
If a Gene (`?j`) is associated with a Target Gene (`?b`), and a Drug (`?e`) is associated with `?j`, but that Drug is **Contraindicated** for Disease (`?a`), then Disease (`?a`) is associated with Target (`?b`).
**Conclusion:** If a drug makes a disease worse (contraindicated), the biological pathways and genes that drug interacts with must be fundamentally involved in the disease's pathology. It infers disease mechanisms based on adverse drug reactions.

**Example:**
If the drug *Benztropine* (`?e`) is contraindicated for *Alzheimer's Disease* (`?a`), and *Benztropine* (`?e`) blocks the *CHRM1* receptor (`?j`), and *CHRM1* (`?j`) interacts with the *APP* gene (`?b`), then *Alzheimer's Disease* (`?a`) is associated with the *APP* gene (`?b`).

---

## 3. Ontological Inheritance (Child to Parent)
**Rule:**
`IF (?f ASSOCIATED_WITH ?b) AND (?a PARENT ?f) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 87.9%
* **Support:** 24,600

**Meaning & Conclusion:**
If a specific sub-disease or child-entity (`?f`) is associated with a Gene (`?b`), then the broader parent category (`?a`) is also associated with that Gene.
**Conclusion:** Properties cascade up the hierarchy. 

**Example:**
If *Early-Onset Alzheimer's* (`?f`) is associated with the *PSEN1* gene (`?b`), and *Alzheimer's Disease* (`?a`) is the parent category of *Early-Onset Alzheimer's* (`?f`), then *Alzheimer's Disease* (`?a`) is also associated with the *PSEN1* gene (`?b`).

---

## 4. Ontological Inheritance (Parent to Child)
**Rule:**
`IF (?e ASSOCIATED_WITH ?b) AND (?e PARENT ?a) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 15.7%
* **Support:** 83,506

**Meaning & Conclusion:**
If a broad parent category (`?e`) is associated with a Gene (`?b`), then the specific child entity (`?a`) is also associated with it.
**Conclusion:** Properties can cascade down the hierarchy, though with much lower confidence (15.7%). Just because a parent category involves a certain gene doesn't mean *every specific sub-type* relies on that exact gene.

**Example:**
If *Dementia* (`?e`) is associated with the *MAPT* gene (`?b`), and *Dementia* (`?e`) is the parent of *Vascular Dementia* (`?a`), then *Vascular Dementia* (`?a`) might be associated with the *MAPT* gene (`?b`) (though with only 15.7% probability).

---

## 5. Hierarchical Phenotype Inference (Variant A)
**Rule:**
`IF (?j ASSOCIATED_WITH ?b) AND (?f PARENT ?j) AND (?a PHENOTYPE_PRESENT ?f) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 99.9%
* **Support:** 1,784

**Meaning & Conclusion:**
If a disease (`?a`) presents a broad phenotype (`?f`), and that phenotype is the parent category of a specific sub-phenotype (`?j`) which is associated with Gene (`?b`), then the disease is associated with Gene (`?b`).
**Conclusion:** Expanding on Rule 1, AMIE learned that phenotypic associations can survive one hop through the ontological hierarchy. Broad symptoms are still reliable indicators of underlying genetic causes.

**Example:**
If *Alzheimer's Disease* (`?a`) presents with *Cognitive Decline* (`?f`), and *Cognitive Decline* (`?f`) is the parent phenotype of *Aphasia* (`?j`), and *Aphasia* (`?j`) is associated with Gene X (`?b`), then *Alzheimer's Disease* (`?a`) is associated with Gene X (`?b`).

---

## 6. Hierarchical Phenotype Inference (Variant B)
**Rule:**
`IF (?j ASSOCIATED_WITH ?b) AND (?a PARENT ?f) AND (?f PHENOTYPE_PRESENT ?j) => THEN (?a ASSOCIATED_WITH ?b)`
* **Confidence:** 99.4%
* **Support:** 5,286

**Meaning & Conclusion:**
If a specific sub-disease (`?f`) presents a phenotype (`?j`) linked to Gene (`?b`), then the parent disease category (`?a`) is also associated with that Gene.
**Conclusion:** A combination of Rules 1 and 3. Genetic associations discovered via phenotypes in a sub-disease are inherited by the parent disease category.

**Example:**
If *Late-Onset Alzheimer's* (`?f`) presents with *Amyloid Plaques* (`?j`), and *Amyloid Plaques* (`?j`) are associated with the *APOE* gene (`?b`), and *Alzheimer's Disease* (`?a`) is the parent of *Late-Onset Alzheimer's* (`?f`), then *Alzheimer's Disease* (`?a`) is associated with the *APOE* gene (`?b`).
