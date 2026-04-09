def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    
    w = (max(values) - min(values)) / num_bins
    if w == 0:
        return [0] * len(values)
    bins = []
    for i in range(len(values)):
        tmp = int((values[i] - min(values))/w)
        bins.append(min(tmp, num_bins-1))
    return bins

    