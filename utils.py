# -*- coding=GBK -*-

import numpy as np
import scipy.linalg
from scipy.stats.stats import spearmanr
import time
import re
import pdb
import sys
import os
import glob
import logging
from psutil import virtual_memory
import os.path
import random
import unicodedata
import sys

unicode_punc_tbl = dict.fromkeys( i for i in xrange(128, sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P') )

logging.basicConfig( level=logging.DEBUG )
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def str2dict(s):
    wordlist = re.split( "\s+", s )
    return dict.fromkeys(wordlist, 1)

stopwordStr = '''a about above across after again against all almost alone along also
although always am among an and another any anybody anyone anything
apart are around as  at away be because been before behind being below
besides between beyond both but by can cannot could  did do does doing done
down  during each either else enough etc  ever every everybody
everyone except far few for  from get gets got had has have having
he her here herself him himself his how however if in indeed instead into
is it its itself just kept me maybe might  more most mostly much must
my myself  neither  no nobody none nor not nothing  of off often on one
only onto or other others ought our ours out  own  please
pp quite rather really said seem  shall she should since so
some somebody somewhat still such than that the their theirs them themselves
then there therefore these they this thorough thoroughly those through thus to
together too toward towards until up upon was we well were what
whatever when whenever where whether which while who whom whose why will with
within would yet you your yourself
re d ll m ve t s'''

stopwordDict = str2dict(stopwordStr)

np.seterr(all="raise")
np.set_printoptions(suppress=True, threshold=np.nan, precision=3)

def initConsoleLogger(loggerName):
    consoleLogger = logging.getLogger(loggerName)
    streamHandler = logging.StreamHandler()
    consoleLogger.addHandler(streamHandler)  
    return consoleLogger
    
def initFileLogger(loggerName, isAppending=False):
    loggerName = os.path.splitext(loggerName)[0]
    currDate = timeToStr( time.time(), "%m.%d" )
    filename = "%s-%s.log" %( loggerName, currDate )
    sn = 0
    while os.path.isfile(filename):
        sn += 1
        filename = "%s-%s-%d.log" %( loggerName, currDate, sn )
  
    fileLogger = logging.getLogger(loggerName)
    if isAppending:
        mode = 'a'
    else:
        mode = 'w'
        
    fileHandler = logging.FileHandler(filename, mode=mode)
    fileLogger.addHandler(fileHandler)
    return fileLogger
    
def warning(*objs):
    sys.stderr.write(*objs)
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.tstart = time.time()
        self.tlast = self.tstart
        self.firstCall = True

    def getElapseTime(self, isStr=True):
        totalElapsed = time.time() - self.tstart
        # elapsed time since last call
        interElapsed = time.time() - self.tlast
        self.tlast = time.time()

        firstCall = self.firstCall
        self.firstCall = False

        if isStr:
            if self.name:
                if firstCall:
                    return '%s elapsed: %.2f' % ( self.name, totalElapsed )
                return '%s elapsed: %.2f/%.2f' % ( self.name, totalElapsed, interElapsed )
            else:
                if firstCall:
                    return 'Elapsed: %.2f' % ( totalElapsed )
                return 'Elapsed: %.2f/%.2f' % ( totalElapsed, interElapsed )
        else:
            return totalElapsed, interElapsed

    def printElapseTime(self):
        print self.getElapseTime()

def timeToStr(timeNum, fmt="%H:%M:%S"):
    timeStr = time.strftime(fmt, time.localtime(timeNum))
    return timeStr

# F-norm of a vector or a matrix
def normF(M, Weight=None):
    if len(M.shape) == 1:
        if Weight is not None:
            # M*M makes all elems positive, and all elems of Weight are nonnegative. So no need to take abs()
            return np.sqrt( np.sum( M * M * Weight ) )
        else:
            return np.sqrt( np.sum( M * M ) )
    
    s = 0
    
    if Weight is not None:
        for i in xrange( len(M) ):
            # row by row calculation. 
            # If doing matrix multiplication, a big temporary matrix will be generated, consuming a lot RAM
            row = M[i] * M[i] * Weight[i]
            s += np.sum(row)
    else:
        for i in xrange( len(M) ):
            row = M[i] * M[i]
            s += np.sum(row)

    return np.sqrt(s)

# normalize a 1-d or 2-d array of nonnegative numbers: 
# keep the original array intact, return a copy of normalized array
# when array is 2d:
# axis=0: normalize columns. axis=1: normalize rows (default)
def normalize(data, axis=1):
    if np.min(data) < 0:
        raise RuntimeError("Negative element in data passed to normalize()")
    if data.ndim == 1:
        return data / np.sum(data)
    if axis == 0:
        s = np.sum(data, axis=0)
        return data / s
    elif axis == 1:
        ss = np.sum(data, axis=1)
        return data / np.tile(ss, (data.shape[1],1)).T
    else:
        raise RuntimeError('function normalize: axis must be 0/1')

# normalize a 1-d or 2-d array of numbers, w.r.t. F-norm
def normalizeF(data, axis=1):
    if data.ndim == 1:
        return data / normF(data) 
    
    data2 = np.copy(data)
    # normalize each column of data
    if axis == 0:
        for i in xrange(data2.shape[1]):
            if normF(data2[:,i]) > 0:
                data2[:,i] /= normF(data2[:,i])
        return data2
    
    # normalize each row of data     
    elif axis == 1:
        norms = np.array( [ normF(x) for x in data2 ] )
        norms[ norms==0 ] = 1
        data2 /= norms[:, None]
    else:
        raise RuntimeError('function normalize: axis must be 0/1')
    return data2
  
def save_matrix_as_text( filename, rowTypeName, T, *extraCols, **kwargs ):
    FMAT = open(filename, "wb")
    print "Save %s matrix into '%s'" %(rowTypeName, filename)
    colSep = kwargs.get("colSep", " ")
    
    K, N = T.shape

    #pdb.set_trace()
    extraColNum = len(extraCols)
    
    FMAT.write( "%d %d %d\n" %( K, N, extraColNum ) )
    for i in xrange(K):
        # if rowNames is provided, print the corresponding row name at the beginning of each line
        line = ""
        for j in xrange(extraColNum):
            col = str( extraCols[j][i] )
            line += col + colSep
        line += "%.5f" %T[i,0]
            
        for j in xrange(1, N):
            line += " %.5f" %T[i,j]
        FMAT.write("%s\n" %line)

    FMAT.close()
    print "%d rows of %s(s) (%d-d each) saved" %( K, rowTypeName, N )
        
# load top maxWordCount words, plus extraWords
def load_embeddings( filename, maxWordCount=-1, extraWords={}, record_skipped=False ):
    FMAT = open(filename)
    warning( "Load embedding text file '%s'\n" %(filename) )
    
    V = []
    word2id = {}
    skippedWords = {}

    vocab = []
    precision = np.float32

    try:
        header = FMAT.readline()
        lineno = 1
        match = re.match( r"(\d+) (\d+)", header)
        if not match:
            raise ValueError(lineno, header)

        vocab_size = int(match.group(1))
        N = int(match.group(2))

        if maxWordCount > 0:
            maxWordCount = min(maxWordCount, vocab_size)
        else:
            maxWordCount = vocab_size

        warning( "Will load embeddings of %d words" %maxWordCount )
        if len(extraWords) > 0:
            warning( ", plus %d extra words" %(len(extraWords)) )
        warning("\n")

        # maxWordCount + len(extraWords) is the maximum num of words.
        # V may contain extra rows that will be removed at the end
        V = np.zeros( (maxWordCount + len(extraWords), N), dtype=precision )
        wid = 0
        orig_wid = 0

        for line in FMAT:
            lineno += 1
            line = line.strip()
            # end of file
            if not line:
                if orig_wid != vocab_size:
                    raise ValueError( lineno, "%d words declared in header, but %d read" %( vocab_size, orig_wid ) )
                break

            fields = line.split(' ')
            # remove empty fields
            fields = filter( lambda x: x, fields )
            w = fields[0]

            if w in extraWords:
                del extraWords[w]
                isInterested = True
            elif orig_wid < maxWordCount:
                isInterested = True
            elif record_skipped:
                isInterested = False
                skippedWords[w] = 1
            else:
                break
							
            orig_wid += 1

            if isInterested:
                V[wid] = np.array( [ float(x) for x in fields[1:] ], dtype=precision )
                word2id[w] = wid
                vocab.append(w)
                wid += 1

            if orig_wid % 1000 == 0:
                warning( "\r%d    %d    %d    \r" %( orig_wid, wid, len(extraWords) ) )

            if orig_wid > vocab_size:
                raise ValueError( "%d words declared in header, but more are read" %(vocab_size) )

    except ValueError, e:
        if len( e.args ) == 2:
            warning( "Unknown line %d:\n%s\n" %( e.args[0], e.args[1] ) )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            warning( "Source line %d - %s on File line %d:\n%s\n" %( tb.tb_lineno, e, lineno, line ) )
        exit(2)

    FMAT.close()
    warning( "\n%d embeddings read, %d kept\n" %(orig_wid, wid) )

    #pdb.set_trace()

    if wid < len(V):
        V = V[:wid]

    # V: embeddings, vocab: array of words, word2id: dict of word to index in V
    return V, vocab, word2id, skippedWords

def loadVocabFile(filename):

    UNI = open(filename)
    vocab_dict = {}
    wid = 1
    
    for line in UNI:
        line = line.strip()
        if line[0] == '#':
            continue
        if filename=='top1grams-wiki.txt':    
            fields = line.split("\t")
            # id, freq, prob
            vocab_dict[ fields[0] ] = ( wid, int(fields[1]), np.exp(float(fields[2])) )
        else:
            vocab_dict[line]=(wid)
        wid += 1
    print "%d words loaded from unigram file %s" %(wid, filename)
    return vocab_dict

