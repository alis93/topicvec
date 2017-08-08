import sys
import pdb
import os
import getopt
from corpusLoader import *
from utils import *
from topicvecDir import topicvecDir
import yaml


from palmettopy.palmetto import Palmetto
palmetto = Palmetto()


def usage():
    print """Usage: modify or create your own yml file to change configurations
        example usage >> python topicExp.py config.yml
    """



try:
    opts, args = getopt.getopt(sys.argv[1:], "i:t:wso")

    if len(args) == 0:
        raise getopt.GetoptError("Not enough free arguments")
    if len(args) > 1:
        raise getopt.GetoptError("Too many free arguments")
    yml_file_path = args[0]

except getopt.GetoptError, e:
    print e.msg
    usage()
    sys.exit(2)



with open(yml_file_path, 'r') as ymlfile:
    config = yaml.load(ymlfile)

# ymlcfg = cfg.items()
# pythoncfg = config.items()

# print len(ymlcfg)
# print len(pythoncfg)

# for i in xrange(len(ymlcfg)):
#     print("from yaml : " + ymlcfg[i][0] + ':' + str(ymlcfg[i][1]))
#     print("from code : " + pythoncfg[i][0] + ':' + str(pythoncfg[i][1]) )



if not config['output_folder_path'].endswith('/'):
    config['output_folder_path']+='/'


output_file = config['output_folder_path']+config['output_file_name']


config['logfilename'] = output_file
topicvec = topicvecDir(**config)
out = topicvec.genOutputter(0)

subsetDocNum, orig_docs_words, orig_docs_name = load_docs(config['corpus_file'])

basename = "%s-%d" % (output_file, subsetDocNum)

docs_idx = topicvec.setDocs(orig_docs_words, orig_docs_name)

readDocNum = len(docs_idx)
out("%d docs left after filtering empty docs" % (readDocNum))
assert readDocNum == topicvec.D, "Returned %d doc idx != %d docs in Topicvec" % (readDocNum, topicvec.D)

# infer topics from docs, and save topics and their proportions in each doc

best_last_Ts, Em, docs_Em, Pi = topicvec.inference()

best_it, best_T, best_loglike = best_last_Ts[0]
last_it, last_T, last_loglike = best_last_Ts[1]

save_matrix_as_text(basename + "-em%d-best.topic.vec" % best_it, "best topics", best_T)
save_matrix_as_text(basename + "-em%d-last.topic.vec" % last_it, "last topics", last_T)
