# Copyright 2022 Olga Kolotuhina, Juan L. Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Function to run GIES with the modified score
# (scores.exp_gauss_int_l0_pen) which does not remove the term
# depending on the intervention targets; the original implementation
# for GIES does this because in their setting the intervention targets
# are fixed

library(pcalg)
path_out = '/tmp/gies_R_test/'
data = read.csv('/tmp/gies_R_test/data.csv', header = FALSE)
targets <- list(integer(0), 2, 3)
target.index <- c(rep(1, 10000), rep(2, 10000), rep(3, 10000))

score = new("GaussL0penIntScore", data, targets, target.index, intercept = TRUE)
gies.fit <- gies(score)

A <- matrix(0, ncol(data), ncol(data))
A[1, 3] <- 1
A[2, 3] <- 1
A[3, 4] <- 1
A[3, 5] <- 1
A[4, 5] <- 1

A_0 <- matrix(0, ncol(data), ncol(data))

score_empty_graph <- score$global.score(as(A_0, "GaussParDAG"))
score$global.score(as(A_0, "GaussParDAG"))
score_gies <- score$global.score(gies.fit$repr)
score_true <- score$global.score(as(A, "GaussParDAG"))
score$pp.dat$scatter[[score$pp.dat$scatter.index[1]]]/score$pp.dat$data.count[1]


scores <- c(score_empty_graph, score_gies, score_true)
write.table(scores,'/tmp/gies_R_test/scores.csv',row.names=FALSE,col.names=FALSE)

P <- matrix(0, ncol(data), ncol(data))
for (k in 1:ncol(data))
{
  nodes = gies.fit$essgraph$.in.edges[[k]]
    for (node in nodes)
    {
      P[node, k] <- 1
    }
}

write.table(P, '/tmp/gies_R_test/A_gies.csv',row.names=FALSE,col.names=FALSE)
