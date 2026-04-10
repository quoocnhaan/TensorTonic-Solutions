def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[0:k]
    
    common = list(set(top_k) & set(relevant))

    precision = len(common) / k
    recall = len(common) / len(relevant)

    return list([precision, recall])