import string
import subprocess
import re

# generate the 318 4probe_[id].txt files from methylation data
clinical_string = subprocess.check_output('cut -f 1 ../dat/y_data.txt',shell=True)
clinical_list = clinical_string.split()

tcga_meth_dir = '/Users/sakellarios/Documents/04-MD-PhD/5-Research/TCGA/hnscc/methylation'
methylation_string = subprocess.check_output('ls', cwd=tcga_meth_dir)
methylation_list_1 = methylation_string.split()
methylation_list_2 = [fname.split('__')[2][:12] for fname in methylation_list_1]

#print methylation_list_2

intersect_list = [fname for fname in methylation_list_2 if fname in clinical_list]
intersect_list = list(set(intersect_list))

print intersect_list
print len(intersect_list)

# four HNSCC associated genes are KDM6A, MET, NFE2L2, SOX2
# we will only keep the probes associated with these genes as our feature space





