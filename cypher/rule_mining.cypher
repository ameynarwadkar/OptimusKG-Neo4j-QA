// 8. Frequent 2-hop typed graph patterns (Rule Mining)
MATCH (source)-[r1]->(middle)-[r2]->(target)
WHERE elementId(source) <> elementId(target)
RETURN
    labels(source) AS source_labels,
    type(r1) AS relation_1,
    labels(middle) AS middle_labels,
    type(r2) AS relation_2,
    labels(target) AS target_labels,
    count(*) AS support
ORDER BY support DESC
LIMIT 100;


// 9. The "Golden Path" for Alzheimer's Drug Repurposing
// Based on the top frequent rule: Drug -[INDICATION]-> Disease -[ASSOCIATED_WITH]-> GeneProtein
// This query finds the exact drugs and genetic targets linked to Alzheimer's in your graph.
MATCH (drug:Drug)-[r1:INDICATION]->(disease:Disease)-[r2:ASSOCIATED_WITH]->(gene:GeneProtein)
WHERE toLower(disease.name) CONTAINS "alzheimer"
RETURN drug.name AS Drug_Name,
       disease.name AS Alzheimer_Variant,
       gene.name AS Target_Gene,
       r1.optimus_label AS Drug_Indication_Type,
       r2.optimus_label AS Gene_Association_Type
LIMIT 50;

