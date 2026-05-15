// =====================================================
// OptimusKG Audit Queries
// =====================================================


// 1. Node counts
MATCH (n)
RETURN labels(n) AS labels, count(n) AS count
ORDER BY count DESC;


// 2. Relationship type counts
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC;


// 3. Relationship property counts
MATCH ()-[r]->()
RETURN type(r) AS relation,
       r.optimus_label AS display_relation,
       count(r) AS count
ORDER BY count DESC
LIMIT 100;


// 4. Disease nodes containing Alzheimer
MATCH (d:Disease)
WHERE toLower(d.name) CONTAINS "alzheimer"
RETURN d.id AS id,
       d.name AS disease,
       d.source AS source
LIMIT 30;


// 5. Direct neighbors of Alzheimer disease
MATCH (d:Disease)-[r]-(n)
WHERE toLower(d.name) CONTAINS "alzheimer"
RETURN d.name AS disease,
       type(r) AS relationship_type,
       r.optimus_label AS display_relation,
       labels(n) AS neighbor_labels,
       n.name AS neighbor_name
LIMIT 100;


// 6. Drug-Gene-Disease paths around Alzheimer
MATCH path = (drug:Drug)-[r1]-(g:GeneProtein)-[r2]-(d:Disease)
WHERE toLower(d.name) CONTAINS "alzheimer"
RETURN drug.name AS drug,
       type(r1) AS drug_gene_rel_type,
       r1.optimus_label AS drug_gene_display,
       g.name AS gene,
       type(r2) AS gene_disease_rel_type,
       r2.optimus_label AS gene_disease_display,
       d.name AS disease
LIMIT 100;


// 7. Disease-Gene-Pathway paths around Alzheimer
MATCH path = (d:Disease)-[r1]-(g:GeneProtein)-[r2]-(p:Pathway)
WHERE toLower(d.name) CONTAINS "alzheimer"
RETURN d.name AS disease,
       type(r1) AS disease_gene_rel_type,
       r1.optimus_label AS disease_gene_display,
       g.name AS gene,
       type(r2) AS gene_pathway_rel_type,
       r2.optimus_label AS gene_pathway_display,
       p.name AS pathway
LIMIT 100;
