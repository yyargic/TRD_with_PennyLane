Tensor rank decomposition (TRD) is an NP-hard problem. An example of this problem is this: "What is the most optimal algorithm to compute the product of two N x N matrices which requires the minimum number of multiplications?"

The standard matrix multiplication algorithm uses N^3 products to multiply N x N matrices. Strassen found in 1969 an algorithm for N=2 which uses only 7 products instead of 8, showing that the N^3 bound is not optimal. Since then, the quest for finding more efficient algorithms has been an active research area, and the optimal algorithm for N>=3 is still unknown. Recently in 2022, Google DeepMind's AlphaTensor joined the search.

I am interested in finding out whether TRD is a problem in which quantum computers would have a significant advantage. Part of my past work at Microsoft Research was dedicated to answering this question. I am sharing here two research reports that I wrote at the time.

In the first report, 230113.pdf, I show that Grover's algorithm can be used for TRD on finite fields. I implemented this approach using PennyLane in grover.py in this repository.

In the second report, 230120.pdf, I give an equivalent restatement to TRD in terms of multi-partite entanglement, which indicates that TRD is intrinsically a quantum problem: "Given a fixed, entangled, pure state \ket{\psi} living in the tensor product of three Hilbert spaces, how do we determine the minimum number of non-entangled states for which \ket{\psi} can be written as their superposition?"

This is my unfinished work and unpublished research, and I plan to continue working on it in the future. I think this problem can be approached with quantum machine learning, using entanglement entropy as the cost function. Please contact me if you would like to collaborate.
