# MIT License
#
# Copyright (c) 2018 Diviyan Kalainathan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

library(methods)
library(pcalg)

# >>> json for new regimes
#inst_packages <-  installed.packages()
#if ("rjson" %in% inst_packages[, 1]) {
#     #uninstalls package
#     remove.packages("rjson")
#     #re-installs package
#     install.packages("rjson")
#} else {
#     install.packages("rjson")
#}
library(rjson)
# <<<

dataset <- read.csv(file='{FOLDER}{FILE}', header=FALSE, sep=",");
# >>> add target support
targets <- fromJSON(file='{FOLDER}{FILE2}')
targets_unique <- lapply(targets$targets, as.integer)
targets_index <- as.integer(targets$index)
# <<<

if({SKELETON}){
  fixedGaps <- read.csv(file='{FOLDER}{GAPS}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}else{
  fixedGaps = NULL
}
# >>>
# targets = sorted(set(regimes)) in Python terms
# target.index = np.argsort(regimes)
score <- new("{SCORE}", data = dataset,
             targets = targets_unique,
             target.index = targets_index)
#score <- new("{SCORE}", data = dataset)
# <<<
result <- pcalg::gies(score, fixedGaps=fixedGaps)
gesmat <- as(result$essgraph, "matrix")
gesmat[gesmat] <- 1
  #gesmat[!gesmat] <- 0
write.csv(gesmat, row.names=FALSE, file = '{FOLDER}{OUTPUT}');
