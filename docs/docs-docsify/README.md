# Overview
**X-Solver**, A library of iterative solution methods on 
heterogeneous clusters.

Efficient and accurate solution of large and sparse linear systems
of algebraic equations is crucial for many problems in scientific computing.
X-Solver is motivated to implement a collection of Krylov subspace 
solution methods and preconditioners on distributed memory clusters equipped with GPUs.

Three primary contributions are made and included in X-Solver to 
realize the motivation.   

First, in addition to the conventional solvers and preconditioners, 
we also provide alternatives that either better exploit the new 
characteristics of modern hardware or turn out to be efficient 
for specific problems.    
For example, X-Solver supports the preconditioners based on the 
fine-grained parallel incomplete factorizations and the solution of 
linear systems of saddle-point form which are dominant for a broad 
range of applications.    

Second, to execute on distributed memory systems, we adopt the 
readily available arguments in the existing codes to implement the 
communication associated with the solvers and preconditioners in the 
background. This new functionality provides more friendly support 
for the solution of linear systems arising in the finite discretizations of PDEs.    

Third, utilizing the MPI + OpenMP, CUDA and HIP programming models 
delivers the respective compute backends and enables the portability 
of X-Solver on heterogeneous devices by different vendors. 

# Functionality
To be the middleware between applications and devices, the selection 
of solution and preconditioning methods included in X-Solver concerns 
the following considerations. 

First, the new characteristics of modern architectures play an important 
role in determining the appropriate algorithms for the target hardware. 
For example, the inherent serialization of preconditioners based on 
incomplete factorizations hampers satisfactory performance on many-core 
architectures like GPUs. Fine-grained parallel variants 
[ref] with significantly increased parallelism justify the inclusion 
and further optimization in X-Solver.

In addition, newly fielded GPUs support increasingly faster arithmetic 
operations in lower precisions. This feature favors the algorithms of 
the compensation for the accuracy inefficiency caused by lower-precision 
calculations. Thus, the nested Krylov iterative refinement method in 
mixed precisions is implemented in X-Solver. 

Second, linear systems arising in real-world applications could be very 
challenging for efficient solution methods. The linear system in 
saddle-point form is a typical example and attracts much attention 
due to its widespread presence, see [ref]. 
The starting point of specific preconditioners for saddle-point systems 
is the block LU decomposition. This fact facilitates the block LU 
solvers in X-Solver to form the basis of applying the so-arising preconditioners.       

## X-Solver Abilities
| Functionality		| Description		|
| :---------------: | :---------------: |
| Iterative solvers | MINRES, CG (classic and pipelined), BiCGStab (classic and pipelined), GMRES, Flexible GMRES, GCR, IDR.									|
| Preconditioners 	| conventional and fine-grained parallel incomplete factorizations, algebraic multigrid, sparse approximate inverse, block-Jacobi, block structured preconditioners for saddle-point systems.			|
| Mixed-precision solvers | nested Krylov iterative refinement in mixed precision.					|
| Sparse BLAS		| levels 1, 2, and 3. |
| Sparse matrix formats | compressed sparse row/column (CSR/CSC), coordinate format (COO), block sparse row format (BSR), ELLPACK (ELL), ELL+COO (HYB) .		|
| Backends			| MPI + X (OpenMP, CUDA, HIP)  |

## Support for Distributed-Memory Clusters
X-Solver aims to solve linear systems of equations on distributed memory 
systems, where large-scale scientific and engineering computations are 
conducted. Realization of the preconditioned Krylov subspace methods on 
the target systems needs to consider the associated communication patterns, 
e.g. global reduction and Neighbor-to-Neighbor data exchange.  
The backend of the associated communication patterns via the MPI layer, 
the de facto standard for distributed memory systems, entails a set of 
data-exchange details.   
The data-exchange details can be obtained from the pattern of the 
nonzero entries suppose that rows and columns of the distributed matrices 
are globally numbered. Solver libraries PETSc and PARALUTION adopt 
this approach. However, there is a broad range of applications in the 
scientific computing community where the global indices may be unessential 
and unavailable in the existing codes.   
For instance, in fields of finite discretizations of PDEs, linear systems 
of algebraic equations with only local indices are most often. CFD is a 
typical example. From the implementation viewpoint, local-to-global 
transferring could give the user a substantial amount of programming tasks. 
To circumvent this difficulty, X-Solver provides a complementary 
functionality, i.e., xsolver\_communicator to collect the readily available 
arguments, instead of the global indices, for the implementation of the 
associated communication in the background. A schematic diagram of the new 
functionality is shown in [Figure 1-1](#fig11).

<figure style="max-width: 350px; margin: 0 auto;">
  <img id="fig11" src="/gh-pages/docs/docs-docsify/images/newfunc.png" width="350" height="300" alt="xsolver impression"/>
  <figcaption>
    Figure 1-1 X-Solver support for interprocessor communication on a cluster 
  </figcaption>
</figure>

## Portability
X-Solver achieves portability to different architectures by implementing 
the respective compute backends, which are summarized as follows.

* MPI + OpenMP &#x2010; designed for x86 and ARM CPUs,
* MPI + CUDA &#x2010; designed for NVIDIA GPUs,
* MPI + HIP &#2010; designed for AMD GPUs and Hygon DCUs.

[Figure 1-2](#fig12) schematically illustrates the procedure of 
solving a linear system by X-Solver. As seen, APIs supplied by X-Solver 
exhibit the following features. 

First, the code snippet in [Figure 1-2](#fig12) remains unchanged 
when executed with either of the compute backends. This offers an easy 
exploit of the computational power of various devices.   
Second, no solid programming skills are acquired from the user due to 
the hidden translation of the code snippet to lower-level OpenMP, HIP, 
or CUDA codes.  
Last but not the least, the utilized prescribe-execute API template 
allows an easy combination of various solvers and preconditioners. 
The optimal choice can be effectively determined. 
 
<figure style="max-width: 600px; margin: 0 auto;">
  <img id="fig12" src="/gh-pages/docs/docs-docsify/images/xsolverv2-impression.png" width="600" height="400" alt="xsolver impression"/>
  <figcaption>
    Figure 1-2 An illustration of solving linear system with X-Solver
  </figcaption>
</figure>

# Architecture and Models

All Krylov subspace methods are combinations of the preconditioning 
application and a reduced set of sparse BLAS routines. This fact means 
that the compute backends of the key sparse BLAS routines and the 
setup and application of preconditioners are crucial for the execution 
on the targeting systems. Compute backends consist of the implementation 
by the respective programming models, the optimization by the algorithm 
improvements and hardware-specific characteristics, the realization of 
the associated communication, and the cross-linking between the above aspects.
From the implementation viewpoint, it is convenient to classify them into 
different modules, which yields the layered architecture of X-Solver, 
as shown in [Figure 2-1](#fig21)</a>.

<figure style="max-width: 600px; margin: 0 auto;">
  <img id="fig21" src="/gh-pages/docs/docs-docsify/images/xsolver-abstraction.png" width="600" height="600" alt="xsolver abstraction"/>
  <figcaption>
    Figure 2-1 Schematic overview of X-Solver's architecture
  </figcaption>
</figure>

## Krylov Subspace Methods

X-Solver exploits Krylov subspace methods to solve the large and sparse 
linear system, denoted by 
<math display="inline">
  <mrow>
    <mi>A</mi>
    <mi>x</mi>
	<mo>=</mo>
	<mi>b</mi>
  </mrow>
</math>
. According to the property of the coefficient matrix 
<math display="inline">
  <mrow>
    <mi>A</mi>
  </mrow>
</math>
, an appropriate solution method is employed.  
[Table 2-1](#tb21) shows the applicability of the available solution 
methods in X-Solver.  

Krylov subspace solution methods rely on a reduced set of algebraic 
operations: sparse matrix-vector (SpMV) multiplication, a linear 
combination of vectors (VecComb), and dot product (DOT). 
Except for VecComb, the other two algebraic operations require data 
communication when executed on distributed memory systems. Namely, 
global reduction operations among all engaged processes are performed 
by dot products. Neighbor-to-Neighbor data exchanges are necessary for 
SpMV multiplications. The number of 
algebraic operations and the associated communication is summarized in 
[Table 2-2](#tb22).   
The preconditioner-dependent algebraic operations 
and communication patterns exclude the summarization of preconditioners 
in [Table 2-2](#tb22).

In Tables [2-1](#tb21) and [2-2](#tb22), the terms SGS and MGS 
denote the standard and modified Gram-Schmidt orthogonalization processes, 
respectively. The modified Gram-Schmidt variant is numerically more robust 
than the standard one. On the other hand, the modified Gram-Schmidt 
variant is more expensive. FGMRES denotes the flexible generalized minimum 
residual (GMRES) method. Like the generalized conjugate residual (GCR) method, 
FGMRES also allows variable preconditioners. 

<table id="tb21" style="margin-left: 50px;">
  <caption>
    Table 2-1 Applicability of the available solution methods in X-Solver
  </caption>
  <thead>
    <tr>
	  <th colspan="4">Coefficient matrix 
	    <math display="inline">
		  <mrow><mi><strong>A</strong></mi></mrow>
		</math>
	  </th>
	</tr>
    <tr>
	  <th></th>
	  <th>Symmetric Positive Definite</th>
	  <th>Symmetric Indefinite</th>
	  <th>Non-Symmetric</th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>CG [r]</td>
	  <td style="text-align: center">&#x2605;</td>
	  <td></td>
	  <td></td>
	</tr>
    <tr>
	  <td>pipelinedCG [r]</td>
	  <td style="text-align: center">&#x2605;</td>
	  <td></td>
	  <td></td>
	</tr>
    <tr>
	  <td>MINRES</td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	  <td></td>
	</tr>
    <tr>
	  <td>BiCGStab [r]</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>pipelinedBiCGStab [r]</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>GCR-SGS [r]</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>GCR-MGS</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>GMRES-SGS [r]</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>GMRES-MGS</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>FGMRES-SGS [r]</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>FGMRES-MGS</td>
	  <td></td>
	  <td></td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
    <tr>
	  <td>Nested Krylov IR</td>
	  <td style="text-align: center">&#x2605;</td>
	  <td style="text-align: center">&#x2605;</td>
	  <td style="text-align: center">&#x2605;</td>
	</tr>
  </tbody>
</table>

<table id="tb22" style="margin-left: auto; margin-right: auto; margin-top: 30px;">
  <caption>
    Table 2-2 At each iteration the maximum number of the algebraic operations 
	and the associated communication in the available solution methods. The term 
	<math display="inline">
	  <mrow><mi>n</mi></mrow>
	</math>
	 denotes the size of unknowns. GCR and GMRES solvers are restarted after 
	<math display="inline">
	  <mrow><mi>m</mi></mrow>
	</math>
	 iterations.
  </caption>
  <thead>
    <tr>
	  <th></th>
	  <th colspan="3">Computation</th>
	  <th colspan="2">Communication</th>
	  <th>Storage</th>
	</tr>
    <tr>
	  <th></th>
	  <th>VecComb</th>
	  <th>SpMV</th>
	  <th>DOT</th>
	  <th>Neighbour-to-Neighbour</th>
	  <th>Global reduction</th>
	  <th></th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>CG</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>4</mn><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>pipelinedCG</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>4</mn><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>MINRES</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>4</mn><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>BiCGStab</td>
	  <td style="text-align: center">4</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">4</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">3</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>7</mn><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>pipelinedBiCGStab</td>
	  <td style="text-align: center">6</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">6</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>9</mn><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>GCR-SGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi>
		  <mo>+</mo><mn>2</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>GCR-MGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi>
		  <mo>+</mo><mn>2</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi></mrow>
		</math>
	  </td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi><mi>n</mi></mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>GMRES-SGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi><mi>n</mi>
		  <mo>+</mo><msup><mi>m</mi><mn>2</mn></msup>
		  </mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>GMRES-MGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi><mi>n</mi>
		  <mo>+</mo><msup><mi>m</mi><mn>2</mn></msup>
		  </mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>FGMRES-SGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">2</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi><mi>n</mi>
		  <mo>+</mo><msup><mi>m</mi><mn>2</mn></msup>
		  </mrow>
		</math>
	  </td>
	</tr>
    <tr>
	  <td>FGMRES-MGS</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">1</td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mi>m</mi>
		  <mo>+</mo><mn>1</mn></mrow>
		</math>
	  </td>
	  <td style="text-align: center">
	    <math display="inline">
		  <mrow><mn>2</mn><mi>m</mi><mi>n</mi>
		  <mo>+</mo><msup><mi>m</mi><mn>2</mn></msup>
		  </mrow>
		</math>
	  </td>
	</tr>
  </tbody>
</table>

### Nested Krylov Iterative Refinement in Mixed Precision
Iterative refinement (IR) is a traditional method for improving 
an approximate solution to a linear system 
<math display="inline">
  <mrow><mi>A</mi><mi>x</mi>
  <mo>=</mo><mi>b</mi></mrow>
</math>
 by repeating the following steps as necessary
+ computing the residual 
<math display="inline">
  <mrow><mi>r</mi><mo>=</mo>
  <mi>b</mi><mo>-</mo><mi>A</mi><mi>x</mi></mrow>
</math>
+ solving the correction equation 
<math display="inline">
  <mrow><mi>A</mi><mi>d</mi>
  <mo>=</mo><mi>r</mi></mrow>
</math>
+ updating the solution 
<math display="inline">
  <mrow><mi>x</mi><mo>=</mo>
  <mi>x</mi><mo>+</mo><mi>d</mi></mrow>
</math>

IR algorithm attracts renewed interest because on modern computer 
architectures lower precision arithmetic is usually faster than 
higher precision arithmetic. For example, single-precision arithmetic 
is usually at least twice as fast as double-precision arithmetic. 
Furthermore, half-precision (16-bit) floating-point arithmetic 
is now available with a further speedup and proportional energy consumption savings.  

Motivated to exploit the improved performance of modern architectures, 
it is practicable to perform the heaviest arithmetic of the IR method 
at lower precisions, i.e., approximately solving the correction equation. 
The utilization of higher precisions in residual computation and solution 
update tries to achieve the final solution at the desired accuracy.   
Thus, in X-Solver we employ the preconditioned Krylov subspace methods at 
lower precisions to solve the correction equation. We refer to the so-arising 
method as the nested Krylov iterative refinement (NKIR) algorithm in 
mixed precision. The NKIR method is sketched in [Algorithm 1](#alg1) and 
contains three precisions:

- <math display="inline"><msub><mi>u</mi><mi>w</mi></msub></math> is the 
working precision at which residuals and updates are calculated, 
and the data <math display="inline"><mrow><mi>A</mi></mrow></math>, 
<math display="inline"><mrow><mi>b</mi></mrow></math> and the solution 
<math display="inline"><mrow><mi>x</mi></mrow></math> are stored, 
- <math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math> is the 
precision at which Krylov subspace methods are applied to solve the correction equations,
- <math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math> is the 
precision at which preconditioners are set up and preconditioning operations are performed. 

The precisions are assumed to satisfy
<math display="block">
  <mrow>
    <msub><mi>u</mi><mi>p</mi></msub>
	<mo>&#x2264;</mo>
    <msub><mi>u</mi><mi>k</mi></msub>
	<mo>&#x2264;</mo>
    <msub><mi>u</mi><mi>w</mi></msub>
  </mrow>
</math>

and three precisions are supported, i.e., double, single and half precisions.
The choice 
<math display="inline">
  <mrow>
    <msub><mi>u</mi><mi>p</mi></msub>
	<mo>=</mo>
    <msub><mi>u</mi><mi>k</mi></msub>
	<mo>=</mo>
    <msub><mi>u</mi><mi>w</mi></msub>
  </mrow>
</math>
 reduces to the fixed precision NKIR algorithm. 

<div class="algorithm" id="alg1">
  <hr />
  <p id="alg_nm"><strong>Algorithm 1</strong> Nested Krylov iterative refinement in mixed precision</p>
  <hr />
  <p>
    1: Given the initial solution 
	  <math display="inline"><msub><mi>x</mi><mn>0</mn></msub></math>
	<br />
	2: <strong>for</strong>
	  <math display="inline"><mi>k</mi><mo>=</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mo>&#8230;</mo>
	  </math> <strong>do</strong>
	<br />
	3: <span class="tab"></span><math display="inline"><msub><mi>r</mi><mi>k</mi></msub><mo>=</mo><mi>b</mi>
	    <mo>-</mo><mi>A</mi><msub><mi>x</mi><mrow><mi>k</mi><mo>-</mo><mn>1</mn></mrow></msub>
	  </math>
	  <span style="margin-left: 70px;">at the working precision 
	  <math display="inline"><msub><mi>u</mi><mi>w</mi></msub></math></span>
	<br />
	4: <span class="tab"></span>Solve
	  <math display="inline"><msup><mi>P</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mi>A</mi>
	    <msub><mi>d</mi><mi>k</mi></msub><mo>=</mo><msup><mi>P</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
		<msub><mi>r</mi><mi>k</mi>
	  </math>
	  <span style="margin-left: 30px;">KSP at precision <math display="inline">
	   <msub><mi>u</mi><mi>k</mi></msub></math> and 
	   precondition <math display="inline"><msup><mi>P</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup></math>
	    at precision <math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>
	  </span>
	<br />
	5: <span class="tab"></span>    
	  <math display="inline"><msub><mi>x</mi><mi>k</mi></msub><mo>=</mo>
	    <msub><mi>x</mi><mrow><mi>k</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
	    <msub><mi>d</mi><mi>k</mi></msub>
	  </math>
	  <span style="margin-left: 70px;">at precision 
	  <math display="inline"><msub><mi>u</mi><mi>w</mi></msub></math></span>
	<br />
	6: <span class="tab">check convergence</span>
	<br />
	7: <strong>end for</strong>
  </p>
  <hr />
</div>

### Pipelined Krylov Subspace Methods
Contrary to Neighbor-to-Neighbor data exchanges, global reduction 
operations could lead to significant decay of scalability with 
large numbers of engaged processors on certain types of clusters. 
This is due to the utilized tree-like hierarchical algorithms.  
Motivated to reduce the number of global reduction operations, 
we develop and implement a pipelined conjugate gradient (pipelinedCG)
method and a pipelined stabilized biconjugate gradient (PipelinedBiCGStab) 
method. The pipelined alternatives are mathematically equivalent to the 
classic methods, i.e., calculating the same solution [ref].  
For easy comparison, we use different colors to distinguish the algebraic 
operations in Algorithms [3](#alg3) and [2](#alg2). The overhead of 
the pipelined solvers is a slight increase in algebraic operations, 
see [Table 2-2](#tb22).

<div class="algorithm">
  <hr />
  <p><strong>Algorithm 2</strong> The classic and pipelined BiCGStab methods</p>
  <hr />

  <div class="alg_grid" id="alg2">
    <div class="alg_l">
      <p style="margin-top: 12ex;"><strong>The classic BiCGStab:</strong></p>
	  <p style="margin-left: 5px;">
	    1: <math display="inline"><msub><mi>r</mi><mn>0</mn></msub><mo>=</mo>
		<mi>b</mi><mo>-</mo><mi>A</mi><msub><mi>x</mi><mn>0</mn></msub></math>
		<br />
		2: <math display="inline"><msub><mi>&#x3c1;</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>&#x3b1;</mi><mn>0</mn></msub><mo>=</mo><msub><mi>w</mi><mn>0</mn></msub>
		<mo>=</mo><mn>1</mn><mo>,</mo><msub><mi>v</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>p</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn></math>
		<br />
		3: <strong>for </strong><math display="inline"><mi>n</mi><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		4: <span class="tab" style="color: blue"><math display="inline">
		<msub><mi>&#x3c1;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>r</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn>
		</mrow></msub></math>
		</span>
		<br />
		5: <span class="tab"><math display="inline">
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3c1;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>&#x3c1;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		</mfrac>
		<mfrac><mrow><msub><mi>&#x3b1;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		<mrow><msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		</mfrac>
		</math></span>
		<br />
		6: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>p</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub><mo>(</mo>
		<msub><mi>p</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		<msub><mi>v</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>)</mo>
		</math></span>
		<br />
		7: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>v</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		8: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3c1;</mi><mi>n</mi></msub></mrow>
		<mrow><msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>v</mi><mi>n</mi></msub>
		</mfrac></math></span>
		<br />
		9: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>s</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><msub><mi>v</mi><mi>n</mi></msub>
		</math></span>
		<br />
		10: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>t</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>s</mi><mi>n</mi></msub>
		</math></span>
		<br />
		11: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>w</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msubsup><mi>t</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>s</mi><mi>n</mi></msub></mrow>
		<mrow><msubsup><mi>t</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>t</mi><mi>n</mi></msub></mrow>
		</mfrac>
		</math></span>
		<br />
		12: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>x</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>x</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><msub><mi>p</mi><mi>n</mi></msub><mo>+</mo>
		<msub><mi>w</mi><mi>n</mi></msub><msub><mi>s</mi><mi>n</mi></msub>
		</math></span>
		<br />
		13: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>r</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>s</mi><mi>n</mi></msub><mo>-</mo>
		<msub><mi>w</mi><mi>n</mi></msub><msub><mi>t</mi><mi>n</mi></msub>
		</math></span>
		<br />
		14: <strong>end for</strong>
		<br />
	  </p>
    </div>
    <div class="alg_r">
      <p><strong>The pipelined BiCGStab</strong></p>
	  <p>
	    1: <math display="inline">
		<msub><mi>r</mi><mn>0</mn></msub><mo>=</mo>
		<mi>b</mi><mo>-</mo><mi>A</mi><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>p</mi><mn>0</mn></msub><mo>=</mo>
		<mi>A</mi><mi>A</mi><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>t</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>p</mi><mn>0</mn></msub><mo>+</mo><mi>A</mi><msub><mi>r</mi><mn>0</mn></msub><mo>,</mo>
		</math><br /><math display="inline" style="margin-left: 1em;">
		<msub><mi>&#x3c3;</mi><mn>0</mn></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>t</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>&#x3c6;</mi><mn>0</mn></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>r</mi><mn>0</mn></msub>
		</math>
		<br />
		2: <math display="inline">
		<msub><mi>&#x3c0;</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>&#x3b3;</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>m</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn><mo>,</mo>
		<msub><mi>q</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>v</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>z</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn><mo>,</mo>
		</math><br /><math display="inline" style="margin-left: 1em;">
		<msub><mi>&#x3c1;</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>&#x3b1;</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>w</mi><mn>0</mn></msub><mo>=</mo><mn>1</mn>
		</math>
		<br />
		3: <strong>for </strong><math display="inline"><mi>n</mi><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		4: <span class="tab"><math display="inline">
		<msub><mi>&#x3c1;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		<msub><mi>m</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		</math></span>
		<br />
		5: <span class="tab"><math display="inline">
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3c1;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>&#x3c1;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		</mfrac><msub><mi>&#x3b1;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		<mo>,</mo>
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3b4;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		</mfrac>
		</math></span>
		<br />
		6: <span class="tab"><math display="inline">
		<msub><mi>&#x3b3;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c3;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub>
		<msub><mi>&#x3b3;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub>
		<msub><mi>&#x3c0;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		</math></span>
		<br />
		7: <span class="tab"><math display="inline">
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3c1;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>&#x3b4;</mi><mi>n</mi></msub>
		</mfrac></math></span>
		<br />
		8: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>v</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>t</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		<msub><mi>p</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub>
		<msub><mi>v</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub>
		<msub><mi>q</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		</math></span>
		<br />
		9: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>q</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>v</mi><mi>n</mi></msub>
		</math></span>
		<br />
		10: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>s</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><msub><mi>v</mi><mi>n</mi></msub>
		</math></span>
		<br />
		11: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>t</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>t</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>w</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		<msub><mi>p</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>q</mi><mi>n</mi></msub>
		</math></span>
		<br />
		12: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>z</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<mfrac><mrow><msub><mi>&#x3b2;</mi><mi>n</mi></msub><msub><mi>&#x3b1;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>&#x3b1;</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow>
		</mfrac>
		<msub><mi>z</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>-</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub>
		<msub><mi>v</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub>
		</math></span>
		<br />
		13: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>p</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>t</mi><mi>n</mi></msub>
		</math></span>
		<br />
		14: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>s</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>&#x3c0;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>q</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>&#x3b8;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>s</mi><mi>n</mi><mi>T</mi></msubsup><msub><mi>t</mi><mi>n</mi></msub>
		</math></span>
		<br />
		15: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>k</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>t</mi><mi>n</mi><mi>T</mi></msubsup><msub><mi>t</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>m</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>t</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>&#x3b7;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup><msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		16: <span class="tab"><math display="inline">
		<msub><mi>w</mi><mi>n</mi></msub><mo>=</mo>
		<mfrac><mrow><msub><mi>&#x3b8;</mi><mi>n</mi></msub></mrow>
		<mrow><msub><mi>k</mi><mi>n</mi></msub>
		</mfrac></math></span>
		<br />
		17: <span class="tab"><math display="inline">
		<msub><mi>&#x3c3;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>m</mi><mi>n</mi></msub><mo>-</mo>
		<msub><mi>w</mi><mi>n</mi></msub>
		<msub><mi>&#x3b7;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		18: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>x</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>x</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>z</mi><mi>n</mi></msub><mo>+</mo>
		<msub><mi>w</mi><mi>n</mi></msub>
		<msub><mi>s</mi><mi>n</mi></msub>
		</math></span>
		<br />
		19: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>r</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>s</mi><mi>n</mi></msub><mo>-</mo>
		<msub><mi>w</mi><mi>n</mi></msub><msub><mi>t</mi><mi>n</mi></msub>
		</math></span>
		<br />
		20: <strong>end for</strong>
		<br />
	  </p>
    </div>
  </div>
  <hr />
</div>

<div class="algorithm">
  <hr />
  <p><strong>Algorithm 3</strong> The classic and pipelined CG methods</p>
  <hr />

  <div class="alg_grid" id="alg3">
    <div class="alg_l">
      <p><strong>The classic CG:</strong></p>
	  <p style="margin-left: 5px;">
		1: <math display="inline">
		<msub><mi>r</mi><mn>0</mn></msub><mo>=</mo>
		<mi>b</mi><mo>-</mo><mi>A</mi><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>p</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>r</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>&#x3c6;</mi><mn>0</mn></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup>
		<msub><mi>r</mi><mn>0</mn>
		</math>
		<br />
		2: <strong>for </strong><math display="inline"><mi>n</mi><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		3: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>q</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		4: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3c3;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>q</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		5: <span class="tab"><math display="inline">
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub><mo>&#x2215;</mo>
		<msub><mi>&#x3c3;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		6: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>x</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>x</mi><mi>n</mi></msub><mo>+</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		7: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>r</mi><mi>n</mi></msub><mo>-</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>q</mi><mi>n</mi></msub>
		</math></span>
		<br />
		8: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3c6;</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msubsup><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow><mi>T</mi></msubsup>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub>
		</math></span>
		<br />
		9: <span class="tab"><math display="inline">
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>&#x2215;</mo>
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		10: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>p</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		11: <strong>end for</strong>
		<br />
	  </p>
    </div>
    <div class="alg_r">
      <p><strong>The pipelined CG</strong></p>
	  <p style="margin-left: 5px;">
		1: <math display="inline">
		<msub><mi>r</mi><mn>0</mn></msub><mo>=</mo>
		<mi>b</mi><mo>-</mo><mi>A</mi><msub><mi>x</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>p</mi><mn>0</mn></msub><mo>=</mo>
		<msub><mi>r</mi><mn>0</mn></msub><mo>,</mo>
		<msub><mi>&#x3c6;</mi><mn>0</mn></msub><mo>=</mo>
		<msubsup><mi>r</mi><mn>0</mn><mi>T</mi></msubsup>
		<msub><mi>r</mi><mn>0</mn>
		</math>
		<br />
		2: <strong>for </strong><math display="inline"><mi>n</mi><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		3: <span class="tab" style="color: red;"><math display="inline">
		<msub><mi>q</mi><mi>n</mi></msub><mo>=</mo>
		<mi>A</mi><msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		4: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3c3;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>q</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>p</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>&#x3c9;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>r</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>q</mi><mi>n</mi></msub><mo>,</mo>
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub><mo>=</mo>
		<msubsup><mi>q</mi><mi>n</mi><mi>T</mi></msubsup>
		<msub><mi>q</mi><mi>n</mi></msub>
		</math></span>
		<br />
		5: <span class="tab"><math display="inline">
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub><mo>&#x2215;</mo>
		<msub><mi>&#x3c3;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		6: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>x</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>x</mi><mi>n</mi></msub><mo>+</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		7: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>r</mi><mi>n</mi></msub><mo>-</mo>
		<msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>q</mi><mi>n</mi></msub>
		</math></span>
		<br />
		8: <span class="tab" style="color: blue;"><math display="inline">
		<msub><mi>&#x3c6;</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub><mo>-</mo>
		<mn>2</mn><msub><mi>&#x3b1;</mi><mi>n</mi></msub>
		<msub><mi>&#x3c9;</mi><mi>n</mi></msub><mo>+</mo>
		<msubsup><mi>&#x3b1;</mi><mi>n</mi><mn>2</mn></msubsup>
		<msub><mi>&#x3b4;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		9: <span class="tab"><math display="inline">
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub><mo>=</mo>
		<msub><mi>&#x3c6;</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>&#x2215;</mo>
		<msub><mi>&#x3c6;</mi><mi>n</mi></msub>
		</math></span>
		<br />
		10: <span class="tab" style="color: green;"><math display="inline">
		<msub><mi>p</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo>
		<msub><mi>r</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>+</mo>
		<msub><mi>&#x3b2;</mi><mi>n</mi></msub>
		<msub><mi>p</mi><mi>n</mi></msub>
		</math></span>
		<br />
		11: <strong>end for</strong>
		<br />
	  </p>
    </div>
  </div>
  <hr />
</div>

### Summary of APIs in the Solution Module  
Currently, the Krylov subspace solution module KSP contains the following APIs:

<dl>
  <dt>xsolver_ksp_settype</dt>
  <dd>Prescribe the KSP solution method from <a href="#/?id=tb21">Table 2-1</a>.</dd>

  <dt>xsolver_ksp_setoptions</dt>
  <dd>Setup the arguments in the KSP methods, e.g. the 
	maximum number of iterations, the relative and absolute stopping tolerances, 
	the initial guess for the solution vector and the restart iterations (if necessary).
	Output the arguments and vectors, e.g. the used iterations and the vector containing 
	the residual at each iteration.
  </dd>

  <dt>xsolver_NKIR_settype</dt>
  <dd>Prescribe the KSP solution and preconditioning methods in 
	the NKIR algorithm. Arguments in the KSP solution and preconditioning methods are 
	prescribed via the APIs in the respective modules.
  </dd>
  <dt>xsolver_NKIR_setprecisions</dt>
  <dd>Prescribe the precisions in the mixed precision NKIR algorithm.</dd>
</dl>

## Preconditioners
Mathematically efficient and numerically cheap preconditioners are crucial 
for the efficiency and robustness of Krylov subspace solution methods. 
In X-Solver the provided preconditioners are summarized in [Table 2-3](#tb23) 
and grouped into algebraic and problem-specific preconditioners. 

+ **Algebraic preconditioners** only rely on the coefficient matrix itself for the setup. 
	X-Solver includes the following preconditioners in this classification.  
	1. Incomplete factorizations:
		<ol type="A">
		  <li>incomplete LU and Cholesky factorizations with zero fill-ins and dual-threshold [r].</li>
		  <li>diagonal compensations of the incomplete factorizations [r].</li>
		  <li>fine-grained parallel algorithms for computing incomplete factorizations [r].</li>
		</ol>
	2. Sparse approximate inverse preconditioners
		<ol type="A">
		  <li>based on the Frobenius norm minimization with a priori sparsity pattern 
		  and a post-filtering technique [r].</li>
		  <li>based on the Sherman-Morrison algorithm [r].</li>
		</ol>
	3. Algebraic multigrid preconditioners:

		BoomerAMG from the Hypre library [r]. 

	4. Block Jacobi preconditioners:

		the aforementioned preconditioners wrapped in a block-Jacobi type.

+ **Problem-specific preconditioners** rely on not only the coefficient matrix 
	but also the underlying information, e.g. structure of the coefficient matrix.
	Currently, X-Solver implements block structured preconditioners with the 
	algebraic or user-supplied Schur complement approximations for saddle point systems [r]. 

<table id="tb23" style="margin-left: 40px;">
  <caption>
    Table 2-3 Preconditioning algorithms in X-Solver
  </caption>
  <thead>
    <tr>
	  <th>Classification</th>
	  <th>Symbol</th>
	  <th>Description</th>
	</tr>
  </thead>
  <tbody>
    <tr><td colspan="3"><strong>Algebriac preconditioners</strong></td></tr>
	<tr>
	  <td>Incomplete</td>
	  <td>ILU0 &#38; MILU0</td>
	  <td>zero fill-ins and the modified variant by diagonal compensation.</td>
	<tr>
	<tr>
	  <td>LU</td>
	  <td>ILUT</td>
	  <td>dual-threshold: dropping tolerance and maximum fill-ins per row.</td>
	</tr>
	<tr>
	  <td>factorizations</td>
	  <td>parILU0 &#38; parILUT</td>
	  <td>fine-grained parallel implementations of ILU0 and ILUT.</td>
	</tr>
	<tr>
	  <td>Imcomplete</td>
	  <td>ICH0 &#38; MICH0</td>
	  <td>zero fill-ins and the modified variant by diagonal compensation.</td>
	</tr>
	<tr>
	  <td>Cholesky</td>
	  <td>ICHT</td>
	  <td>dual-threshold: dropping tolerance and maximum fill-ins per row.</td>
	</tr>
	<tr>
	  <td>factorizations</td>
	  <td>parICH0 &#38; parICHT</td>
	  <td>fine-grained parallel implementations of ICH0 and ICHT.</td>
	</tr>
	<tr>
	  <td>Approximate inverse</td>
	  <td>SPAILS</td>
	  <td>based on the least-squares (Frobenius norm) minimization.</td>
	</tr>
	<tr>
	  <td>preconditioners</td>
	  <td>SPAISM</td>
	  <td>based on the Sherman-Morrison algorithm.</td>
	</tr>
	<tr>
	  <td>Algebraic multigrid <br />preconditioner</td>
	  <td>BoomerAMG</td>
	  <td>algebraic multigrid preconditioner from the Hypre library.</td>
	</tr>
	<tr>
	  <td colspan="3"><strong>Problem-specific preconditioners</strong></td>
	</tr>
	<tr>
	  <td>Block structured preconditioner</td>
	  <td>BSTRUC</td>
	  <td>specific for saddle-point systems.</td>
	</tr>
  </tbody>
</table>

### Fine-grained Parallel Incomplete Factorization Preconditioners
Preconditioners based on incomplete factorizations are widely utilized 
in scientific computing. However, fine-grained parallelism is unavailable 
for the preconditioner setup and application in the conventional formulations 
of incomplete factorizations. Motivated to exploit the computing power 
of many-core architectures, we adopt the fine-grained parallel incomplete 
LU factorizations proposed in [ref]. As seen from [Algorithm 4](#alg4), 
each nonzero element of the incomplete factors L and U on the priori sparsity 
pattern is computed by iteration sweeps that are performed in parallel.  
Thus, the algorithm is easy to parallelize and has much more parallelism 
than the existing approaches which mainly rely on reordering the matrix to 
enhance the parallelism.  
[Algorithm 4](#alg4) performs the incomplete factorization with a priori 
sparsity pattern often the same as the input matrix and thus is denoted by parILU0. 
Improvements by dynamically updating the sparsity pattern via evaluating 
the residual norm 
<math display="inline">
  <mo>&#x2225;</mo><mi>A</mi><mo>-</mo>
  <mi>L</mi><mi>U</mi><mo>&#x2225;</mo>
</math>
 during the iteration steps are developed in [ref], see [Algorithm 4](#alg4).  
The so-arising factorization parILUT is akin to threshold-based ILU and could 
produce more efficient preconditioners for difficult problems.    

CUDA and HIP compute backends of the fine-grained parallel factorization 
algorithms for the nonsymmetric and symmetric positive definite matrices are 
included in X-Solver to support NVIDIA and AMD GPUs. In addition, for the 
time-dominant computation of the residual matrix 
<math display="inline">
  <mi>A</mi><mo>-</mo><mi>L</mi><mi>U</mi>
</math>
 in the parILUT factorization, the proposed register-aware algorithm of 
sparse matrix-matrix multiplications significantly improves the performance 
of parILUT than the existing libraries.

Applying ILU preconditioners in a parallel environment needs to enhance the parallelism 
of sparse triangular solvers. In X-Solver, we apply a fixed small number 
of Jacobi iterations to solve a triangular system on GPUs, contrary to 
the methods based on level scheduling.

<div class="algorithm">
  <hr />
  <p><strong>Algorithm 4</strong> Fine-grained parallel incomplete factorization algorithms</p>
  <hr />

  <div class="alg_grid" id="alg4">
    <div class="alg_l">
      <p><strong>parILU0:</strong></p>
	  <p><strong>Input: </strong>initial values of 
	    <math display="inline"><msub><mi>l</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></math>
		 and 
	    <math display="inline"><msub><mi>u</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></math>
		 on the priori sparsity 
	    <math display="inline"><mi>S</mi></math>
	  </p>
	  <p><strong>Output: </strong>the incomplete factors 
	    <math display="inline"><mi>L</mi></math>
		 and 
	    <math display="inline"><mi>U</mi></math>
	  </p>
	  <p style="margin-left: 5px;">
		1: <strong>for </strong>sweep <math display="inline"><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		2. <span class="tab"><strong>for </strong><math display="inline"><mo>(</mo>
		<mi>i</mi><mo>,</mo><mi>j</mi><mo>)</mo><mo>&#x2208;</mo><mi>S</mi></math>
		<strong> parallel do</strong></span>
		<br />
		3. <span class="tab"></span><span class="tab"></span><strong>if </strong>
		<math display="inline"><mi>i</mi><mo>></mo><mi>j</mi></math>
		<strong> then</strong>
		<br />
		4. <span class="tab"></span><span class="tab"></span><span class="tab"></span>
	    <math display="inline"><msub><mi>l</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo>
		<mo>(</mo><msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>-</mo>
		<munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow>
		<mrow><mi>j</mi><mo>-</mo><mn>1</mn></mrow></munderover>
		<msub><mi>l</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub>
		<msub><mi>u</mi><mrow><mi>k</mi><mi>j</mi></mrow></msub><mo>)</mo>
		<mo>&#2215;</mo><msub><mi>u</mi><mrow><mi>j</mi><mi>j</mi></mrow></msub>
		</math>
		<br />
		5. <span class="tab"></span><span class="tab"></span><strong>else </strong>
		<br />
		6. <span class="tab"></span><span class="tab"></span><span class="tab"></span>
	    <math display="inline"><msub><mi>u</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo>
		<msub><mi>a</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>-</mo>
		<munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow>
		<mrow><mi>i</mi><mo>-</mo><mn>1</mn></mrow></munderover>
		<msub><mi>l</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub>
		<msub><mi>u</mi><mrow><mi>k</mi><mi>j</mi></mrow></msub><mo>)</mo>
		</math>
		<br />
		7. <span class="tab"></span><span class="tab"></span><strong>else if </strong>
		<br />
		8. <span class="tab"></span><strong>end for</strong>
		<br />
		9. <strong>end for</strong>
		<br />
	  </p>
	</div>
	<div class="alg_r">
      <p><strong>parILUT:</strong></p>
	  <p><strong>Input: </strong>initial values of 
	    <math display="inline"><mi>L</mi></math>
		 and 
	    <math display="inline"><mi>U</mi></math>
		 factors
	  </p>
	  <p><strong>Output: </strong>the incomplete factors 
	    <math display="inline"><mi>L</mi></math>
		 and 
	    <math display="inline"><mi>U</mi></math>
	  </p>
	  <p style="margin-left: 5px;">
		1: <strong>for </strong>step <math display="inline"><mo>=</mo><mn>1</mn><mo>,</mo>
		<mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mo>&#8230;</mo></math>
		, until convergence <strong>do</strong>
		<br />
		2: <span class="tab"></span>Identify the candidates as the nonzeros of the 
		residual matrix 
		<math display="inline"><mi>R</mi><mo>=</mo><mi>A</mi><mo>-</mo><mi>L</mi><mi>U</mi></math>
		<br />
		3: <span class="tab"></span>Estimate the residual norm 
		<math display="inline"><mo>&#x2225;</mo><mi>A</mi><mo>-</mo>
		<mi>L</mi><mi>U</mi><mo>&#x2225;</mo></math>
		<br />
		4: <span class="tab"></span>Add 
		<math display="inline"><msub><mi>m</mi><mi>L</mi></msub></math>
		 and 
		<math display="inline"><msub><mi>m</mi><mi>U</mi></msub></math>
		 candidates to 
		<math display="inline"><mi>L</mi></math>
		 and 
		<math display="inline"><mi>U</mi></math>
		<br />
		5: <span class="tab"></span>Do one sweep fo the parILU0 algorithm
		<br /> 
		6: <span class="tab"></span>Remove the 
		<math display="inline"><msub><mi>m</mi><mi>L</mi></msub></math>
		 and 
		<math display="inline"><msub><mi>m</mi><mi>U</mi></msub></math>
		 smallest magnitude elements from  
		<math display="inline"><mi>L</mi></math>
		 and 
		<math display="inline"><mi>U</mi></math>
		<br />
		7: <span class="tab"></span>Do one sweep fo the parILU0 algorithm
		<br /> 
		8: <strong>end for</strong>
		<br />
	  </p>
	</div>
  </div>
  <hr />
</div>

### Block Structured Preconditioners
In recent years, a large amount of work has been devoted to the problem
of solving large linear systems in saddle point form as ([2.1](#m21)). 
The reason for this interest is the fact that such problems arise in a wide 
variety of technical and scientific applications. For example, the incompressible 
fluid mechanism has been a major source of saddle point systems. 

X-Solver prefers iterative solution methods to solve saddle point systems, 
which generally are large and sparse. Efficient preconditioners for saddle 
point systems are proposed based on the block LU factorization ([2.2](#m22)). 
The so-arising preconditioners are referred to as block structured 
preconditioners ([2.3](#m23)). The most challenging task is to find 
the numerically cheap and mathematically equivalent approximations of the 
Schur complement, denoted by 
<math display="inline">
  <mover accent="false"><mi>S</mi><mo>&#126;</mo></mover>
</math>
 in ([2.3](#m23)).
 
<math id="m21" display="block">
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>B</mi><mn>2</mn></msub></mtd>
      <mtd><mo>-</mo><mi>C</mi></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>x</mi></mtd>
    </mtr>
    <mtr>
      <mtd><mi>y</mi></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>f</mi></mtd>
    </mtr>
    <mtr>
      <mtd><mi>g</mi></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mtext style="margin-left: 1em;">, or </mtext>
  <mi mathvariant="script" style="margin-left: 1em;">A</mi><mi>u</mi><mo>=</mo><mi>b</mi>
  <mtext>, </mtext>
  <mtext class="eqn_label">(2.1)</mtext>
</math>
<span style="display: block; margin-top: 15px; margin-bottom: 15px;">where</span>
<math display="block">
  <mi>A</mi><mo>&#x2208;</mo>
  <msup><mi mathvariant="script">R</mi><mrow><mi>n</mi><mo>&#xd7;</mo><mi>n</mi></mrow></msup>
  <mtext>, </mtext>
  <msub style="margin-left: 1em;"><mi>B</mi><mn>1</mn></msub><mo>,</mo><msub><mi>B</mi><mn>2</mn></msub>
  <mo>&#x2208;</mo>
  <msup><mi mathvariant="script">R</mi><mrow><mi>m</mi><mo>&#xd7;</mo><mi>n</mi></mrow></msup>
  <mtext>, </mtext>
  <mi style="margin-left: 1em;">C</mi><mo>&#x2208;</mo>
  <msup><mi mathvariant="script">R</mi><mrow><mi>m</mi><mo>&#xd7;</mo><mi>m</mi></mrow></msup>
  <mtext style="margin-left: 1em;">, with</mtext>
  <mi style="margin-left: 1em;">n</mi><mo>></mo><mi>m</mi>
  <mtext>.</mtext>
</math>

<math id="m22" display="block" style="margin-top: 3ex;">
  <mi mathvariant="script">A</mi><mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>B</mi><mn>2</mn></msub></mtd>
      <mtd><mo>-</mo><mi>C</mi></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><mi>O</mi></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>B</mi><mn>2</mn></msub></mtd>
      <mtd><mi>S</mi></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><msub><mi>I</mi><mn>1</mn></msub></mtd>
      <mtd><msup><mi>A</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
	  <msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup></mtd>
    </mtr>
    <mtr>
      <mtd><mi>O</mi></mtd>
      <mtd><msub><mi>I</mi><mn>2</mn></msub></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mtext> , </mtext>
  <mi style="margin-left: 1em;">S</mi><mo>=</mo>
  <mo>-</mo><mo>(</mo><mi>C</mi><mo>+</mo>
  <msub><mi>B</mi><mn>2</mn></msub><msup><mi>A</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup>
  <msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup>
  <mo>)</mo>
  <mtext>.</mtext>
  <mtext class="eqn_label">(2.2)</mtext>
</math>

<math id="m23" display="block" style="margin-top: 3ex;">
  <msub><mi mathvariant="script">P</mi><mrow><mi>B</mi><mi>F</mi></mrow></msub>
  <mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><mi>O</mi></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>B</mi><mn>2</mn></msub></mtd>
      <mtd><mover accent="false"><mi>S</mi><mo>&#126;</mo></mover></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><msub><mi>I</mi><mn>1</mn></msub></mtd>
      <mtd><msup><mrow><mover accent="false"><mi>A</mi><mo>&#126;</mo></mover></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup>
	  <msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup></mtd>
    </mtr>
    <mtr>
      <mtd><mi>O</mi></mtd>
      <mtd><msub><mi>I</mi><mn>2</mn></msub></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mtext> , </mtext>
  <msub style="margin-left: 1em;"><mi mathvariant="script">P</mi><mrow><mi>B</mi><mi>L</mi></mrow></msub>
  <mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><mi>O</mi></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>B</mi><mn>2</mn></msub></mtd>
      <mtd><mover accent="false"><mi>S</mi><mo>&#126;</mo></mover></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mtext> , </mtext>
  <msub style="margin-left: 1em;"><mi mathvariant="script">P</mi><mrow><mi>B</mi><mi>U</mi></mrow></msub>
  <mo>=</mo>
  <mo>[</mo>
  <mtable>
    <mtr>
      <mtd><mi>A</mi></mtd>
      <mtd><msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup></mtd>
    </mtr>
    <mtr>
      <mtd><mi>O</mi></mtd>
      <mtd><mover accent="false"><mi>S</mi><mo>&#126;</mo></mover></mtd>
    </mtr>
  </mtable>
  <mo>]</mo>
  <mtext> . </mtext>
  <mtext class="eqn_label">(2.3)</mtext>
</math>

Interest in block structured preconditioners has appeared in a wide 
variety of academic and industrial applications [ref], which 
justifies the inclusion in X-Solver. 
From the viewpoint of implementation, the common basis of applying 
the block structured preconditioners is the solution of block LU factors. 
The block LU solution consists of solving the subsystems with 
<math display="inline"><mi>A</mi></math>
 and 
<math display="inline"><mover accent="false"><mi>S</mi><mo>&#126;</mo></mover></math>
, where iterative solution methods and algebraic preconditioners included 
in X-Solver are applicable here. 
This feature facilitates block structured preconditioners with the 
following features.

* User merely needs to provide the submatrices and subvectors of the 
  saddle point system [2.1](#m21) instead of the assembled ones. 
* User determines the typical type of the block structured 
  preconditioners [2.3](#m21) and prescribes the solution methods 
  for the involved subsystems. 
* User-supplied Schur complement approximations are allowed. 
  In addition, algebraic approximations are provided by X-Solver, 
  e.g. the default one 
  <math display="inline"><mover accent="false"><mi>S</mi><mo>&#126;</mo></mover>
    <mo>=</mo><mo>-</mo><mi>C</mi>
  </math>
   and 
  the approximation from SIMPLE preconditioner 
  <math display="inline">
    <mover accent="false"><mi>S</mi><mo>&#126;</mo></mover><mo>=</mo>
    <mo>-</mo><mo>(</mo><mi>C</mi><mo>+</mo>
    <msub><mi>B</mi><mn>2</mn></msub><mtext>Diag</mtext><mo>(</mo><mi>A</mi>
	<msup><mo>)</mo><mrow><mo>-</mo><mn>1</mn></mrow></msup>
    <msubsup><mi>B</mi><mn>1</mn><mi>T</mi></msubsup>
    <mo>)</mo>
  </math>
   [ref].

Currently, the block structured preconditioners are in the experimental stage and are planned to be released soon. 

### Summary of APIs in the Preconditioning Module
Currently, the preconditioning module PC contains the following APIs:

<dl>
  <dt>xsolver_ksp_settype</dt>
  <dd>Prescribe the preconditioner from <a href="#/?id=tb23">Table 2-3</a>.</dd>

  <dt>xsolver_pc_setilu</dt>
  <dd>Setup the arguments in the threshold-based incomplete 
	factorizing preconditioners, e.g. the maximum fill-ins and dropping tolerance. 
  </dd>

  <dt>xsolver_pc_setparilu</dt>
  <dd>Setup the arguments in the fine-grained parallel 
	incomplete factorizations, e.g. the number of iteration sweeps and the maximum fill-ins.
  </dd>

  <dt>xsolver_pc_settrisolver</dt>
  <dd>Prescribe the solution method for the sparse 
	triangular systems, i.e. the conventional method (by default) and the 
	Jacobi iterative method with a prescribed number of iterations.
  </dd>

  <dt>xsolver_pc_setspai</dt>
  <dd>Setup the arguments in the sparse approximate inverse preconditioners, e.g. 
  </dd>

  <dt>xsolver_pc_setamg</dt>
  <dd>Setup the arguments in the algebraic multigrid preconditioners, e.g.
  </dd>

  <dt>xsolver_pc_setbstruc</dt>
  <dd>Prescribe the typical type of block structured 
	preconditioners, i.e. 'BL', 'BU' or 'BF' and the solution and preconditioning 
	methods for the involved subsystems. Arguments in the KSP method and 
	preconditioner are prescribed via the AIPs in the respective modules. 
  </dd>

  <dt>xsolver_pc_setschur</dt>
  <dd>Prescribe the Schur complement approximation from the 
	choices 'SIMPLE', 'Default' and 'Specific'. By 'Specific', the user provides 
	the Schur approximation matrix in CSR format. 
  </dd>
</dl>

## Sparse BLAS
X-Solver implements the in-house sparse BLAS of levels 1-3 and this 
section addresses the proposed optimization concerning the sparse 
matrix-matrix (SpGEMM) and matrix-vector (SpMV) multiplications. 
As introduced in previous sections, the SpGEMM kernel is crucial 
for the setup of the fine-grained parallel incomplete factorization 
preconditioners and the SpMV kernel is included in all Krylov subspace solution methods. 

### Sparse Matrix-Matrix Multiplication
Several SpGEMM algorithms have been proposed in recent years for 
many-core processors such as GPUs. However, their implementations 
mostly use low-speed on-chip shared memory and global memory. 
High-speed registers are seriously underutilized. Motivated by 
the underuse of registers in recent SpGEMM algorithms, we propose 
register-aware algorithms to improve the performance of SpGEMM. 
Specifically, we use registers to fetch data and various in-register 
communication schemes to finish computations. In addition, 
the N-to-M product-thread mapping method is adopted to achieve the 
load balance.  
Compared with the existing libraries in [Table 2-4](#tb24), 
our library manages to use registers and shared memory to implement the 
GPU backends of SpGEMM. 
The proposed method requires classifying the rows into several groups 
based on the number of nonzero elements. 
The classification is originally performed on CPUs and it is unsuitable 
for applications performing SpGEMM with dynamic sparsity patterns on GPUs, 
like the fine-grained parallel incomplete factorizations in [Algorithm 4](#alg4). 
To circumvent this challenge, we extend the classification to GPUs.
We refer to [ref] for more discussions of the register-aware SpGEMM algorithms. 
In this work's evaluation section, we illustrate the acceleration of forming 
the fine-grained parallel incomplete factorizations by using the 
register-aware SpGEMM algorithms.

<table id="tb24" style="margin-left: 15px;">
  <caption>Table 2-4 Comparison between different SpGEMM libraries</caption>
  <thead>
    <tr>
	  <th></th>
	  <th>intermediate space allocation</th>
	  <th>global load balancing</th>
	  <th>nonzero-to-thread mapping</th>
	  <th>nonzero compression</th>
	  <th>memory usage</th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>CUSP [ref]</td>
	  <td>upper bound</td>
	  <td>row+all sort</td>
	  <td>1-to-1</td>
	  <td>seg.sum</td>
	  <td>smem + gmem</td>
	</tr>
	<tr>
 	  <td>cuSPARSE [ref]</td>
	  <td>precise</td>
	  <td>row</td>
	  <td>1-to-1</td>
	  <td>hash</td>
	  <td>smem + gmem</td>
	</tr>
	<tr>
	  <td>bhSPARSE [ref]</td>
	  <td>progressive</td>
	  <td>row+bin/cta</td>
	  <td>1-to-1</td>
	  <td>seg.sum + hori.merge</td>
	  <td>smem</td>
	</tr>
	<tr>
	  <td>RMerge [ref]</td>
	  <td>precise</td>
	  <td>row</td>
	  <td>1-to-1</td>
	  <td>vert.merge</td>
	  <td>reg. + gmem</td>
	</tr>
	<tr>
	  <td>X-Solver</td>
	  <td>precise</td>
	  <td>row+bin/cta</td>
	  <td>N-to-M</td>
	  <td>vert.merge</td>
	  <td>reg. + gmem</td>
	</td>
  </tbody>
</table>

### Sparse Matrix-Vector Multiplication
Superposition between computation and communication in sparse 
matrix-vector multiplication is considered by X-Solver. Overlapping 
the halo components' exchange with the computation on the inner 
components can somewhat eliminate the data-transferring overhead. 
As illustrated in [Figure 2-2](#fig22) this strategy leads to the 
pipelined SpMV implementation.  
The number of the halo components is normally much smaller than 
that of the inner components because it only represents the coupling 
of the compute mesh at the borders of the neighboring subdomains. 
In this sense, it is possible to hide at least partially the time 
on uploading the halo components and shuttling the data back 
and forth between the hosts (CPUs) and devices (GPUs). 
_The sparse storage formats supported by X-Solver should be addressed here._

<figure style="max-width: 800px; margin: 0 auto;">
  <img id="fig22" src="/gh-pages/docs/docs-docsify/images/spmv_overlap.png" width="800" height="380" alt="SpMV implementation" />
  <figcaption>
    Figure 2-2 Schematic representation of the pipelined SpMV implementation
  </figcaption>
</figure>

## X-Solver Communicator
This section addresses how to adopt the readily available arguments 
in the existing codes to implement the associated communication 
patterns in Krylov subspace methods.  
The utilized storage scheme for distributed matrices will be 
introduced first, which forms the basic concepts of the new functionality. 

### Distributed Matrix Storage Scheme
In the context of finite discretizations of PDEs on distributed 
memory systems, the global domain is decomposed into a number 
of subdomains to which the partitioned compute meshes are assigned. 
The choice of partitioning methods is tightly related to data 
communication between subdomains. Graph partitioning schemes are 
in common use in the scientific computing community. 
We refer to [ref] for a detailed description of the available 
partitioning methods and their restrictions. 

In [Figure 2-3](#fig23) we take the edge-based partitioning 
as an example to illustrate the following categorization of the 
elements in the distributed matrix and vector. 

<dl>
  <dt>interior</dt>
  <dd>components are those belonging to the subdomain and consist of 
      the inner and interface components;
  </dd>
  <dt>non-interior</dt>
  <dd>components are those belonging to other subdomains;</dd>
  <dt>inner</dt>
  <dd>components are interior components that are coupled only with 
      other interior components;
  </dd>
  <dt>interface</dt>
  <dd>components are interior components that are coupled with non-interior components;</dd>
  <dt>halo (ghost)</dt>
  <dd>components are non-interior components that are coupled with interface components.</dd>
</dl>

<div id="fig23" style="display: flex; flex-direction: row; flex-basis: auto; flex-grow: 1; margin-left: 30px;">
  <figure style="max-width: 500px; border: none;">
    <img src="/gh-pages/docs/docs-docsify/images/decomposition.png" width="450" height="200" alt="decomposion" />
	<figcaption style="background: none;">(a) Compute meshes</figcaption>
  </figure>
  <figure style="max-width: 300px; border: none;">
    <img src="/gh-pages/docs/docs-docsify/images/decomposition2.png" width="250" height="200" alt="decomposion2" />
	<figcaption style="background: none;">(b) Distributed matrix and vector</figcaption>
  </figure>
</div>
<div style="margin-top: -15px;">
  <figcaption>Figure 2-3 Local compute meshes, matrix and vector mapped to one subdomain
  </figcaption>
</div>

The distributed matrix and vector sizes are Interior&#xd7;(Interiori&#x2b;Halo) 
and Interior, respectively.
In X-Solver the interior components are ordered first and followed 
by the halo ones, as seen in [Figure 2-3(b)](#fig23). 
Contrary to global numbering, local numbering, i.e. the row and 
column indices are numbered from zero, is a natural choice for 
distributed matrices arising in finite discretizations of PDEs. 
Therefore, local indices of distributed matrices are preferred by X-Solver. 
Different sparse storage formats are supported by X-Solver to 
represent the distributed matrix. 

There exist other storage schemes for distributed matrices. For example, 
PETSc and Chronos utilize the blocked scheme, i.e., the distributed 
matrix is divided into diagonal and off-diagonal blocks. The blocked 
scheme allows an easy implementation of the block Jacobi preconditioners. 
However, from the standpoint of utilization, it is a bit more 
cumbersome for the user to adjust the representation of distributed 
matrices to meet the blocked scheme.  

### Communicator Arguments
Krylov subspace solution methods rely on a reduced set of algebraic 
operations: sparse matrix-vector multiplication (SpMV), a linear 
combination of vectors (VecComb), and dot product (DOT). 
When preconditioning is included, additional two operations are required: 
preconditioner setup and preconditioning application. The preconditioning 
application is transferred to solve the linear system with the 
preconditioner. The communication patterns involved in the 
preconditioning application are preconditioner-dependent, which are not 
included in the discussion here. 
Except for VecComb, the other two algebraic operations require data 
exchanges. Namely, global reduction operations among all engaged 
processes are performed by dot products.  
Regarding SpMV, arithmetic operations associated with the interface 
rows require the halo components of the multiplying vector uploaded 
from the neighboring processes, see [Figure 2-3(b)](#fig23). 

X-Solver requests the following communicator arguments to implement 
the associated communication patterns in the background. 

<dl>
  <dt>total_nbs_thisproc<dt>
  <dd>integer value: the total number of neighbors</dd>
  <dt>nbs_thisproc</dt>
  <dd>integer array: the neighbors' MPI ranks</dd>
  <dt>nInterior</dt>
  <dd>integer value: the size of the interior components</dd>
  <dt>nHalo</dt>
  <dd>integer value: the size of the halo components</dd>
  <dt>exchange_ptr</dt>
  <dd>integer array</dd>
  <dt>exchange_displs_proc</dt>
  <dd>integer array</dd>
  <dt style="font-weight: normal;">-- the indices of data sent to the i<sup>th</sup> neighbor are <br />
	<span style="margin-left: 10px;">exchange_ptr(exchange_displs_proc(i) : exchange_displs_proc(i+1)-1).</span>
  </dt><dd></dd>
  <dt style="font-weight: normal;">-- data received from the i<sup>th</sup> neighbor is continuously 
    stored from<br />
	<span style="margin-left: 10px;">(nInterior+exchange_displs_proc(i) : nInterior+exchange_displs_proc(i+1)-1).</span>
  </dt><dd></dd>
</dl>

[Figure 2-4](#fig24) demonstrates the communicator arguments in a simple 
example. Concerning rank0, its neighboring processes are rank1 and rank2. 
Indices of the interior elements sent to the neighbors are 
<math display="inline"><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>}</mo></math>
 and 
<math display="inline"><mo>{</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>}</mo></math>
, respectively. Correspondingly, indices of the halo elements received 
from the neighbors are numbered by 
<math display="inline"><mo>{</mo><mn>4</mn><mo>,</mo><mn>5</mn><mo>}</mo></math>
 and 
<math display="inline"><mo>{</mo><mn>6</mn><mo>,</mo><mn>7</mn><mo>}</mo></math>
. These sending and receiving arrangements are crucial to the accomplishment 
of Neighbor-to-Neighbor data exchange by utilizing the send-and-receive MPI routines. 

<figure style="max-width: 800px; margin: 0 auto;">
  <img id="fig24" src="/gh-pages/docs/docs-docsify/images/P2PMPI.png" width="750" height="450" alt="P2PMPI" />
  <figcaption>
    Figure 2-4 An illustration of the communicator arguments. Four subdomains 
	are mapped to the respective MPI processes and arrows represent data 
	exchange associated with rank0.
  </figcaption>
</figure>

In fields of finite discretizations, the communicator arguments belong 
to the mesh topology context and are readily available upon the partition 
and assignment of the compute mesh to subdomains. They are utilized to 
generate the distributed matrix in finite volume methods [ref] or the 
accumulated vector in finite element methods [ref].  

From the viewpoint of utilization, the communicator's arguments exhibit 
the following advantages. First, these communicator arguments are nothing 
but the existing data in the finite discretization context. The user merely 
needs to collect and provide them to X-Solver via the provided API. Second, 
a large number of applications require solving a sequence of linear systems 
of algebraic equations. For instance, time-dependent problems with the 
implicit time integration schemes and nonlinear systems linearized by the 
approximate Newton method. The communicator arguments remain the same in 
these applications, permitting the setup only once within the programming 
procedure. Third, within the scope of the provided preconditioners, these 
communicator arguments are also sufficient to implement the communication 
involved in the preconditioning algorithms apart from the block Jacobi type.     

X-Solver provides the API **xsolver_communicator_setup** to set up 
the communicator arguments.

# Benchmarks and Evaluations
A comprehensive evaluation of all functionalities supplied by X-Solver 
is outside the scope of this work. Instead, evaluations shown in this 
section are meant to illustrate the representative achievements of 
X-Solver, which are described as follows.

* Accelerating the solution of distributed systems of linear equations 
  in real-world applications. To this end, the xsolver4foam package 
  is realized to supply plug-in support for OpenFoam, an open-source 
  CFD simulator [ref] that has been widely utilized in academia and 
  industry. The performance of OpenFoam depends on the efficient solution 
  of large and sparse linear systems of equations arising in the finite 
  volume discretization of PDEs. Evaluations demonstrate the advantage 
  over the built-in solver of OpenFoam and the third-party library PETSc 
  in terms of the reduced end-to-solution time. 
* Exploiting the new characteristics of modern hardware for a broad range 
  of applications. To this end, linear systems with the coefficient 
  matrices from the SuiteSparse Matrix Collection are solved by the 
  two representative solution and preconditioning methods, i.e., 
  preconditioners based on the fine-grained parallel incomplete 
  factorizations on GPUs and the nested Krylov iterative refinement 
  algorithm in mixed precision. Comparisons are performed with the 
  vendors' and open-source libraries, e.g. cuSPARSE and MAGMA.

The utilized benchmarks are first introduced, followed by the description 
of the hardware where evaluations are conducted. 

## Benchmarks

**OpenFoam benchmarks** are briefly described as follows and we refer 
to [ref] for a detailed description.  

<dl>
  <dt>SimpleCar</dt>
  <dd>This benchmark is to simulate the airflow around a sketched car 
    model in 2D. The steady-state Reynolds-averaged Navier-Stokes 
	governing equations are solved by the SIMPLE 
	(Semi-Implicit Method for Pressure Linked Equations) algorithm [ref], 
	consisting of a segregated solution of the momentum and mass 
	conservation equations. At the high Reynolds number 
	<math display="inline"><mi>R</mi><mi>e</mi><mo>=</mo>
	  <msup><mn>10</mn><mn>6</mn></msup>
	</math>
	, the flow becomes turbulent and the 
	<math display="inline"><mi>k</mi><mo>-</mo><mi>&#x3b5;</mi></math>
	 turbulence model is adopted. Two turbulence equations are then 
	solved for the respective turbulence quantities. In summary, 
	each SIMPLE iteration consists of solving two velocity equations, 
	one pressure equation and two turbulence equations.  
  </dd>
  <dt>Lid Driven Cavity</dt>
  <dd>This problem simulates the flow in a square cavity 
	<math display="inline"><msup><mrow><mo>(</mo><mn>0</mn><mo>,</mo>
	<mn>0</mn><mo>.</mo><mn>1</mn><mo>)</mo></mrow><mn>2</mn></msup></math>
	 with enclosed boundary conditions. A lid moves from left to 
	right with a unit horizontal velocity. The reference velocity and length 
	<math display="inline"><mi>U</mi><mo>=</mo><mn>1</mn></math>
	 and 
	<math display="inline"><mi>L</mi><mo>=</mo><mn>0</mn><mo>.</mo><mn>1</mn></math>
	 and the viscosity parameter 
	<math display="inline"><mi>&#x3bd;</mi><mo>=</mo><msup><mn>10</mn>
	<mrow><mo>-</mo><mn>5</mn></mrow></msup></math>
	 result in the Reynolds number 
	<math display="inline"><mi>R</mi><mi>e</mi><mo>=</mo><mi>U</mi>
	<mi>L</mi><mo>&#x2215;</mo><mi>&#x3bd;</mi><mo>=</mo>
	<msup><mn>10</mn><mn>4</mn></msup></math>
	. We utilize the PISO algorithm [ref] for this benchmark to solve the 
	time-dependent laminar Navier-Stokes. 
	The time step 
	<math display="inline"><mi>&#x3b4;</mi><mi>t</mi></math>
	 is set to equal the mesh size 
	<math display="inline"><mi>&#x3b4;</mi><mi>x</mi></math>
	 to ensure the Courant number is less than 
	<math display="inline"><mn>1</mn></math>
	. Thus, temporal accuracy and numerical stability can be achieved when 
	running pisoFoam. In summary, each time step consists of solving two 
	velocity equations and two pressure equations. 
  </dd>
</dl>

**SuiteSparse Matrix Collection** provides sparse matrices from various 
applications. The characteristics of the tested matrices are described in [Table 3-2](#tb32).

## Hardware

<table id="tb31" style="margin-left: 55px;">
  <caption>Table 3-1 Computing hardware configuration</caption>
  <thead>
    <tr>
	  <th></th>
	  <th><strong>x86 CPUs</strong></th>
	  <th><strong>ARM CPUs</strong></th>
	  <th><strong>Nvidia GPUs</strong></th>
	  <th><strong>Hygon DCUs</strong></th>
	</tr>
  </thead>
  <tbody>
	<tr>
	  <td>Architecture</td>
	  <td>Intel 6248R (24&#xd7;2)</td>
	  <td>Kunpeng 920 (48&#xd7;2)</td>
	  <td>Tesla V100s</td>
	  <td>Hygon II</td>
	</tr>
	<tr>
	  <td>Compute Nodes</td>
	  <td>4</td>
	  <td>1</td>
	  <td>1</td>
	  <td>1</td>
	</tr>
	<tr>
	  <td>DP Performance</td>
	  <td>4.6TFLOPS</td>
	  <td>2.0TFLOPS</td>
	  <td>8.2TFLOPS</td>
	  <td>7.0TFLOPS</td>
	</tr>
	<tr>
	  <td>SP Performance</td>
	  <td>9.2TFLOPS</td>
	  <td>4.0TFLOPS</td>
	  <td>16.4TFLOPS</td>
	  <td>14.0TFLOPS</td>
	</tr>
	<tr>
	  <td>Operating Freq.</td>
	  <td>3.0 GHz</td>
	  <td>2.6 GHz</td>
	  <td>-</td>
	  <td>-</td>
	</tr>
	<tr>
	  <td>Mem. Capacity</td>
	  <td>32GB&#xd7;12@2933MHz</td>
	  <td>16GB&#xd7;16@2933MHz</td>
	  <td>32GB</td>
	  <td>16GB</td>
	</tr>
	<tr>
	  <td>Mem. Bandwidth</td>
	  <td>280.8GB/s</td>
	  <td>374.4GB/s</td>
	  <td>1134GB/s</td>
	  <td>1024GB/s</td>
	</tr>
	<tr>
	  <td>PCI-E Bandwidth</td>
	  <td>32Gb/s</td>
	  <td>64Gb/s</td>
	  <td>32Gb/s</td>
	  <td>32Gb/s</td>
	</tr>
	<tr>
	  <td>Infinity Bandwidth</td>
	  <td>100Gb/s</td>
	  <td>-</td>
	  <td>-</td>
	  <td>-</td>
	</tr>
	<tr>
	  <td>Backend of X-Solver</td>
	  <td>MPI + OpenMP</td>
	  <td>MPI + OpenMP</td>
	  <td>MPI + CUDA</td>
	  <td>MPI + HIP</td>
	</tr>
  </tbody>
</table>

<table id="tb32" style="margin-left: 55px; margin-top: 30px;">
  <caption>Table 3-2 Tested matrices from SuiteSparse Matrix Collection</caption>
  <thead>
    <tr>
	  <th>Matrix</th>
	  <th width="120px">n</th>
	  <th width="120px">nnz</th>
	  <th>S.P.D.</th>
	  <th>Description</th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>apache1</td>
	  <td>80800</td>
	  <td>542184</td>
	  <td>yes</td>
	  <td>Structure Problem</td>
	</tr>
    <tr>
	  <td>apache2</td>
	  <td>715176</td>
	  <td>4817870</td>
	  <td>yes</td>
	  <td>Structure Problem</td>
	</tr>
    <tr>
	  <td>cage10</td>
	  <td>11397</td>
	  <td>150645</td>
	  <td>no</td>
	  <td>Directed Weighted Graph</td>
	</tr>
    <tr>
	  <td>cage11</td>
	  <td>39082</td>
	  <td>559722</td>
	  <td>no</td>
	  <td>Directed Weighted Graph</td>
	</tr>
    <tr>
	  <td>thermal1</td>
	  <td>82654</td>
	  <td>574458</td>
	  <td>yes</td>
	  <td>Thermal Problem</td>
	</tr>
    <tr>
	  <td>thermal2</td>
	  <td>1228045</td>
	  <td>8580313</td>
	  <td>yes</td>
	  <td>Thermal Problem</td>
	</tr>
    <tr>
	  <td>thermo_TC</td>
	  <td>102158</td>
	  <td>711558</td>
	  <td>yes</td>
	  <td>Thermal Problem</td>
	</tr>
    <tr>
	  <td>thermo_dM</td>
	  <td>204316</td>
	  <td>1423116</td>
	  <td>yes</td>
	  <td>Thermal Problem</td>
	</tr>
    <tr>
	  <td>tmt_sym</td>
	  <td>726713</td>
	  <td>5080961</td>
	  <td>yes</td>
	  <td>Electromagnetic Problem</td>
	</tr>
    <tr>
	  <td>tmt_unsym</td>
	  <td>917825</td>
	  <td>4584801</td>
	  <td>no</td>
	  <td>Electromagnetic Problem</td>
	</tr>
    <tr>
	  <td>t2em</td>
	  <td>921632</td>
	  <td>4590832</td>
	  <td>no</td>
	  <td>Electromagnetic Problem</td>
	</tr>
    <tr>
	  <td>torso2</td>
	  <td>115967</td>
	  <td>1033473</td>
	  <td>no</td>
	  <td>2D/3D Problem</td>
	</tr>
    <tr>
	  <td>ecology2</td>
	  <td>999999</td>
	  <td>4995991</td>
	  <td>yes</td>
	  <td>2D/3D Problem</td>
	</tr>
    <tr>
	  <td>venkat01</td>
	  <td>62424</td>
	  <td>1717792</td>
	  <td>no</td>
	  <td>Computational Fluid Dynamics</td>
	</tr>
    <tr>
	  <td>parabolic_fem</td>
	  <td>525825</td>
	  <td>3674625</td>
	  <td>yes</td>
	  <td>Computational Fluid Dynamics</td>
	</tr>
    <tr>
	  <td>G3_circuit</td>
	  <td>1585478</td>
	  <td>7660826</td>
	  <td>yes</td>
	  <td>Circuit Simulation</td>
	</tr>
    <tr>
	  <td>memplus</td>
	  <td>17758</td>
	  <td>99147</td>
	  <td>no</td>
	  <td>Circuit Simulation</td>
	</tr>
  </tbody>
</table>

## Evaluations
### xsolver4foam on OpenFoam Benchmarks
The motivation of this section is to demonstrate the capability 
of X-Solver to solve distributed systems of linear equations. 
Supposed a linear system of the explicit coefficient matrix, 
an algebraic way to generate the distributed systems is row-wisely 
dividing the matrix and assigning them to processes. However, 
this method can be far from the generation and solution of distributed 
systems in real-world applications. To fulfill the objective, we 
consider the solution of distributed systems arising in the CFD 
simulator OpenFoam on four computing nodes of Intel CPUs, see [Table 3-1](#tb31).  
The communicator's arguments described in Section 2.4.2 are crucial 
to realize the associated communication patterns in X-Solver. 
As motivated, the arguments are already existing in distributed applications 
that discretize PDEs and solve the so-arising linear systems. Thus, 
the xsolver4foam package collects the existing data in OpenFoam as the 
arguments and in this way glues X-Solver and OpenFoam together. 

Since PETSc also supplies a plug-in petsc4foam for OpenFoam, the 
comparison is carried out between the third-party libraries 
X-Solver and PETSc, and the built-in solver of OpenFoam.  
The three libraries are evaluated in terms of the number of Krylov 
iterations and the nonlinear iterations where appropriate. We choose 
the end-to-solution time using PETSc as the baseline and then calculate 
the respective speedup using the other two libraries. The aforementioned 
results and the prescribed parameters are shown in [Table 3-3](#tb33) 
and [Table 3-4](#tb34), respectively.

As seen, X-Solver delivers a speedup of 2.5x and 4.7x over PETSc on 
the SimpleCar and Lid Driven Cavity benchmarks, respectively. Compared 
with the built-in solver of OpenFoam, the speedup of X-Solver is 1.6x 
and 1.2x. The superiority of X-Solver mainly stems from the utilization 
of the dual-threshold incomplete factorization preconditioner ILUT in 
solving the time-dominant pressure systems. 
It is shown in [Table 3-3](#tb33) that the ILUT preconditioner 
can significantly reduce the number of Krylov iterations. Since the number 
of iterations could vary from nonlinear iterations and time steps, the 
reported results are obtained from a middle stage of nonlinear iterations 
on the SimpleCar case and from the last time step on the Lid Driven Cavity case.  
Despite an increased cost on setup, the application of the ILUT preconditioner 
with a moderate number of fill-ins in the incomplete factors, e.g., 
<math display="inline"><mtext>fillin</mtext><mo>&#x2215;</mo><mtext>row</mtext>
<mo>=</mo><mn>8</mn></math>
, leads to the reduction of the end-to-solution time. OpenFoam formulates 
the ILU preconditioner in a simpler way, which is specific for matrices 
with regular structures rather than for general sparse matrices, see [ref].  
Although the ILUT preconditioner is supported by PETSc, the petsc4foam 
package does not include it. This results in the unavailability of the 
ILUT preconditioner when applying PETSc in OpenFoam. 
For the considered benchmarks the applied preconditioners are in the 
block Jacobi form, where the ILU factorizations are only performed on 
the diagonal blocks. 

<table id="tb33" style="margin-left: 50px;">
  <caption>
    Table 3-3 Evaluations of three solution libraries on OpenFoam benchmarks 
	in terms of the number of iterations and the end-to-solution speedup. 
	4&#xd7;40 MPI processes are utilized on the Intel platform
  </caption>
  <thead>
    <tr>
	  <th></th>
	  <th>Nonlinear_iter</th>
	  <th>V_iter</th>
	  <th>Turb_iter</th>
	  <th>P_iter</th>
	  <th>End-to-solution time</th>
	  <th>Speedup</th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td colspan="7" style="text-align: center;"><strong>SimpleCar</strong></td>
	</tr>
    <tr>
	  <td>PETSc</td>
	  <td>60</td>
	  <td>??</td>
	  <td>??</td>
	  <td>??</td>
	  <td>946s</td>
	  <td>1</td>
	</tr>
    <tr>
	  <td>OpenFoam</td>
	  <td>60</td>
	  <td>??</td>
	  <td>??</td>
	  <td>??</td>
	  <td>594s</td>
	  <td>1.6</td>
	</tr>
    <tr>
	  <td>X-Solver</td>
	  <td>60</td>
	  <td>??</td>
	  <td>??</td>
	  <td>??</td>
	  <td>391s</td>
	  <td>2.5</td>
	</tr>
    <tr>
	  <td colspan="7" style="text-align: center"><strong>Lid Driven Cavity</strong></td>
	</tr>
    <tr>
	  <td>PETSs</td>
	  <td>-</td>
	  <td>??</td>
	  <td>-</td>
	  <td>??</td>
	  <td>165s</td>
	  <td>1</td>
	</tr>
    <tr>
	  <td>OpenFoam</td>
	  <td>-</td>
	  <td>??</td>
	  <td>-</td>
	  <td>??</td>
	  <td>42s</td>
	  <td>3.9</td>
	</tr>
    <tr>
	  <td>X-Solver</td>
	  <td>-</td>
	  <td>??</td>
	  <td>-</td>
	  <td>??</td>
	  <td>35s</td>
	  <td>4.7</td>
	</tr>
  </tbody>
</table>

<table id="tb34" style="margin-left: 75px; margin-top: 30px;">
  <caption>
    Table 3-4 The prescribed parameters for the OpenFoam beanchmarks
  </caption>
  <tbody>
    <tr>
	  <td>Benchmarks</td>
	  <td>Algorithms</td>
	  <td>Solution Parameters</td>
	  <td>Matrix Size</td>
	</tr>
    <tr>
	  <td>SimpleCar</td>
	  <td>SIMPLE</td>
	  <td>nonlinear_tol = <math display="inline"><mn>1.0</mn><mo>&#xd7;</mo>
	    <msup><mn>10</mn><mrow><mo>-</mo><mn>2</mn></mrow></msup></math>
	  </td>
	  <td>
	    <math display="inline"><msup><mn>5176000</mn><mn>2</mn></msup></math>
	  </td>
	</tr>
    <tr>
	  <td>Lid driven cavity</td>
	  <td>PISO</td>
	  <td>
	    <math display="inline"><mi>&#x3b4;</mi><mi>t</mi><mo>=</mo>
	    <mn>6.25</mn><mo>&#xd7;</mo><msup><mn>10</mn><mrow><mo>-</mo><mn>2</mn></mrow></msup></math>
		, 5 time steps
	  </td>
	  <td>
	    <math display="inline"><msup><mn>2560000</mn><mn>2</mn></msup></math>
	  </td>
	</tr>
    <tr>
	  <td><strong>Libraries</strong></td>
	  <td>KSP (All)</td>
	  <td>Preconditioner (P &#x26; Others)</td>
	  <td>linear_tol (P &#x26; Others)</td>
	</tr>
    <tr>
	  <td>PETSc</td>
	  <td>BiCGStab</td>
	  <td>BILU0 &#x26; BILU0</td>
	  <td>1.0e-6 &#x26; 1.0e-4</td>
	</tr>
    <tr>
	  <td>OpenFoam</td>
	  <td>BiCGStab</td>
	  <td>DILU &#x26; DILU</td>
	  <td>1.0e-6 &#x26; 1.0e-4</td>
	</tr>
    <tr>
	  <td>X-Solver</td>
	  <td>BiCGStab</td>
	  <td>BILUT(8, 1e-4) &#x26; BILU0</td>
	  <td>1.0e-6 &#x26; 1.0e-4</td>
	</tr>
  </tbody>
</table>

### Fine-grained Parallel ILUT Preconditioner
X-Solver aims to adopt modern architectures' new characteristics 
to significantly accelerate the solution of linear systems. 
To this end,  X-Solver supplies an attractive preconditioner 
based on the fine-grained parallel factorization with threshold 
parILUT. In this section, we numerically evaluate the preconditioner 
on NVIDIA GPUs described in [Table 3-1](#tb31) for a relatively 
broad range of applications, see the tested matrices in [Table 3-2](#tb32). 

In [Table 3-5](#tb35)  we demonstrate the 
number of GCR iterations and the timings when applying the incomplete 
factors as preconditioners. The comparison is performed between 
parILUT and the conventional factorization with zero fill-ins ilu0 
from the NVIDIA cuSPARSE library. Speedups in terms of the total 
execution time are seen over the cuSPARSE implementation, which 
uses level scheduling to exploit parallelism. Two aspects result in 
this superiority of parILUT, i.e., the better quality of the 
so-arising preconditioner with faster convergence and the much  
more parallelism with dramatically reduced time on the preconditioner's 
setup and application.   

Numerical experiments not included here are utilized to determine 
the appropriate parameters involved in parILUT. In this section, we 
only report the optimal option which leads to a minimal total execution time. 
Five steps are used in [Algorithm 4](#alg4) and the number of fill-ins 
in the incomplete factors is 1.5 times over the input matrix. In the 
total execution time, solving the linear systems with the incomplete 
factor preconditioner, denoted by <em>T<sub>iter_solve</sub></em> 
, is more dominant than building the preconditioner. 
When solving systems with the preconditioner, five Jacobi iterations 
are applied by X-Solver which usually generates a preconditioner 
comparable in quality to the preconditioner computed conventionally. 
Moreover, enhanced parallelism leads to significantly reduced time 
over the conventional triangular solvers that cuSPARSE applies.

The advantage over MAGMA which also includes the parILUT algorithm 
is seen in [Table 3-6](#tb36) . X-Solver achieves 
an average speedup of 14.8x. Excluding the maximal 191.8x speedup on 
the <em>memplus</em> case, the average speedup is 4.4x. The numerical 
results are consistent with the prediction that applying the 
register-aware sparse matrix-matrix multiplications is crucial for 
improving the performance of parILUT.  

Another optimization is to balance the workload by evenly assigning 
the non-zero elements of the incomplete factors to the GPU threads. 
Compared to the assignment by rows as adopted by MAGMA, the 
implementation of X-Solver could deliver a promising speedup on 
the cases with a large variation of non-zeros among rows, see 
the <em>memplus</em> case.
For a fair comparison, the same parameters are prescribed in the 
implementation of parILUT by X-Solver and MAGMA.

Another advantage of the optimized parILUT algorithm is good performance 
portability across different GPU architectures. To illustrate it, the 
relative runtime of the building blocks for the specific problem and 
architecture configuration is shown in [Figure 3-1](#fig31). The building 
blocks correspond to the computing steps involved in parILUT as presented 
in [Algorithm 4](#alg4). Two bars for each problem denote the normalized 
execution time on the NVIDIA V100s GPUs and Hygon DCUs, respectively. 
The results indicate that the relative costs of the distinct building 
blocks seem to be almost independent of GPU architectures. This 
demonstrates the good performance portability of the optimized parILUT algorithm.

<table id="tb35">
  <caption>
    Table 3-5 The number of iterations and execution time (ms) by using 
	the X-Solver's parILUT and cuSPARSE's ilu0 preconditioners, 
	and the speedup of X-Solver over cuSPARSE. GCR iterative 
	solution method of the relative stopping tolerance 1.0e-6 is utilized
  </caption>
  <thead>
    <tr>
	  <th></th>
	  <th colspan="4">parILUT-X-Solver</th>
	  <th colspan="4">ilu0-cuSPARSE</th>
	  <th>Speedup</th>
	</tr>
	<tr>
	  <th>Matrix</th>
	  <th><em>Iter.</em></th>
	  <th><em>T<sub>prec_setup</sub></em></th>
	  <th><em>T<sub>iter_solve</sub></em></th>
	  <th><em>T<sub>total</sub></em></th>
	  <th><em>Iter.</em></th>
	  <th><em>T<sub>prec_setup</sub></em></th>
	  <th><em>T<sub>iter_solve</sub></em></th>
	  <th><em>T<sub>total</sub></em></th>
	  <th></th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>apache1</td>
	  <td>256</td>
	  <td>3.6e+01</td>
	  <td>9.5e+01</td>
	  <td>1.3e+02</td>
	  <td>195</td>
	  <td>1.1e+01</td>
	  <td>5.8e+02</td>
	  <td>5.9e+02</td>
	  <td>4.5</td>
	</tr>
    <tr>
	  <td>apache2</td>
	  <td>584</td>
	  <td>9.6e+01</td>
	  <td>8.1e+02</td>
	  <td>9.0e+02</td>
	  <td>615</td>
	  <td>4.0e+01</td>
	  <td>4.7e+03</td>
	  <td>4.7e+02</td>
	  <td>5.2</td>
	</tr>
    <tr>
	  <td>cage10</td>
	  <td>4</td>
	  <td>2.7e+01</td>
	  <td>1.3e+00</td>
	  <td>2.8e+01</td>
	  <td>4</td>
	  <td>9.4e+00</td>
	  <td>9.2e+00</td>
	  <td>1.9e+01</td>
	  <td>0.7</td>
	</tr>
    <tr>
	  <td>cage11</td>
	  <td>4</td>
	  <td>5.2e+01</td>
	  <td>2.0e+00</td>
	  <td>5.4e+01</td>
	  <td>4</td>
	  <td>9.4e+00</td>
	  <td>1.8e+01</td>
	  <td>2.7e+01</td>
	  <td>0.5</td>
	</tr>
    <tr>
	  <td>thermal1</td>
	  <td>269</td>
	  <td>3.6e+01</td>
	  <td>1.0e+02</td>
	  <td>1.4e+02</td>
	  <td>457</td>
	  <td>1.1e+01</td>
	  <td>3.8e+03</td>
	  <td>3.8e+03</td>
	  <td>27.5</td>
	</tr>
    <tr>
	  <td>thermal2</td>
	  <td>1054</td>
	  <td>1.6e+02</td>
	  <td>2.3e+03</td>
	  <td>2.4e+03</td>
	  <td>1742</td>
	  <td>2.1e+01</td>
	  <td>2.9e+04</td>
	  <td>2.9e+04</td>
	  <td>11.5</td>
	</tr>
    <tr>
	  <td>thermo_TC</td>
	  <td>4</td>
	  <td>4.3e+01</td>
	  <td>2.7e+00</td>
	  <td>4.6e+01</td>
	  <td>6</td>
	  <td>1.0e+01</td>
	  <td>1.5e+01</td>
	  <td>2.6e+01</td>
	  <td>0.6</td>
	</tr>
    <tr>
	  <td>thermo_dM</td>
	  <td>4</td>
	  <td>6.2e+01</td>
	  <td>3.9e+00</td>
	  <td>6.6e+01</td>
	  <td>6</td>
	  <td>1.0e+01</td>
	  <td>2.0e+01</td>
	  <td>3.0e+01</td>
	  <td>0.5</td>
	</tr>
    <tr>
	  <td>tmt_sym</td>
	  <td>617</td>
	  <td>8.7e+01</td>
	  <td>1.4e+03</td>
	  <td>1.5e+03</td>
	  <td>960</td>
	  <td>7.4e+02</td>
	  <td>1.1e+06</td>
	  <td>1.1e+06</td>
	  <td>725.5</td>
	</tr>
    <tr>
	  <td>tmt_unsym</td>
	  <td>1373</td>
	  <td>8.4e+01</td>
	  <td>1.8e+03</td>
	  <td>1.9e+03</td>
	  <td>1317</td>
	  <td>9.1e+02</td>
	  <td>1.8e+06</td>
	  <td>1.8e+06</td>
	  <td>957.5</td>
	</tr>
    <tr>
	  <td>t2em</td>
	  <td>465</td>
	  <td>8.5e+01</td>
	  <td>8.0e+02</td>
	  <td>8.8e+02</td>
	  <td>534</td>
	  <td>8.2e+02</td>
	  <td>6.9e+05</td>
	  <td>6.9e+05</td>
	  <td>785.8</td>
	</tr>
    <tr>
	  <td>torso2</td>
	  <td>4</td>
	  <td>4.4e+01</td>
	  <td>2.8e+00</td>
	  <td>4.7e+01</td>
	  <td>6</td>
	  <td>1.4e+01</td>
	  <td>1.6e+02</td>
	  <td>1.8e+02</td>
	  <td>3.7</td>
	</tr>
    <tr>
	  <td>ecology2</td>
	  <td>1003</td>
	  <td>9.0e+01</td>
	  <td>2.3e+03</td>
	  <td>2.4e+03</td>
	  <td>1483</td>
	  <td>1.2e+02</td>
	  <td>4.3e+04</td>
	  <td>4.3e+04</td>
	  <td>18.1</td>
	</tr>
    <tr>
	  <td>venkat01</td>
	  <td>12</td>
	  <td>5.7e+01</td>
	  <td>1.1e+01</td>
	  <td>6.7e+01</td>
	  <td>16</td>
	  <td>2.0e+01</td>
	  <td>8.0e+02</td>
	  <td>8.2e+02</td>
	  <td>12.1</td>
	</tr>
    <tr>
	  <td>parabolic_fem</td>
	  <td>436</td>
	  <td>8.8e+01</td>
	  <td>4.4e+02</td>
	  <td>5.3e+02</td>
	  <td>1624</td>
	  <td>1.1e+01</td>
	  <td>2.1e+03</td>
	  <td>2.1e+03</td>
	  <td>4.0</td>
	</tr>
    <tr>
	  <td>G3_circuit</td>
	  <td>536</td>
	  <td>1.2e+02</td>
	  <td>1.6e+03</td>
	  <td>1.7e+03</td>
	  <td>579</td>
	  <td>7.8e+01</td>
	  <td>1.9e+04</td>
	  <td>1.9e+04</td>
	  <td>11.5</td>
	</tr>
    <tr>
	  <td>memplus</td>
	  <td>34</td>
	  <td>4.6e+01</td>
	  <td>1.9e+01</td>
	  <td>6.5e+01</td>
	  <td>239</td>
	  <td>1.2e+01</td>
	  <td>6.1e+02</td>
	  <td>6.3e+02</td>
	  <td>9.7</td>
	</tr>
    <tr>
	  <td>csrp</td>
	  <td>75</td>
	  <td>3.8e+01</td>
	  <td>3.3e+01</td>
	  <td>7.2e+01</td>
	  <td>75</td>
	  <td>1.1e+01</td>
	  <td>2.1e+02</td>
	  <td>2.2e+02</td>
	  <td>3.1</td>
	</tr>
  </tbody>
</table>

<table id="tb36" style="margin-left: 75px; margin-top: 30px">
  <caption>
    Table 3-6 Execution time (ms) of X-Solver and MAGMA's implementation of the 
	parILUT factorization and speedup of X-Solver over MAGMA
  </caption>
  <thead>
    <tr>
	  <th>Matrix</th>
	  <th width="150px">X-Solver</th>
	  <th width="150px">MAGMA</th>
	  <th width="150px">Speedup</th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td>apache1</td>
	  <td>3.6e+01</td>
	  <td>8.3e+01</td>
	  <td>2.3</td>
	</tr>
    <tr>
	  <td>apache2</td>
	  <td>9.6e+01</td>
	  <td>3.6e+02</td>
	  <td>3.7</td>
	</tr>
    <tr>
	  <td>cage10</td>
	  <td>2.4e+01</td>
	  <td>1.3e+02</td>
	  <td>5.4</td>
	</tr>
    <tr>
	  <td>cage11</td>
	  <td>4.7e+01</td>
	  <td>2.7e+02</td>
	  <td>5.7</td>
	</tr>
    <tr>
	  <td>thermal1</td>
	  <td>3.6e+01</td>
	  <td>7.7e+01</td>
	  <td>2.2</td>
	</tr>
    <tr>
	  <td>thermal2</td>
	  <td>1.6e+02</td>
	  <td>6.2e+02</td>
	  <td>4.0</td>
	</tr>
    <tr>
	  <td>thermo_TC</td>
	  <td>4.0e+01</td>
	  <td>9.2e+01</td>
	  <td>2.3</td>
	</tr>
    <tr>
	  <td>thermo_dM</td>
	  <td>5.8e+01</td>
	  <td>1.5e+02</td>
	  <td>2.6</td>
	</tr>
    <tr>
	  <td>tmt_sym</td>
	  <td>8.7e+01</td>
	  <td>3.3e+02</td>
	  <td>3.9</td>
	</tr>
    <tr>
	  <td>tmt_unsym</td>
	  <td>8.4e+01</td>
	  <td>2.5e+02</td>
	  <td>3.0</td>
	</tr>
    <tr>
	  <td>t2em</td>
	  <td>8.5e+01</td>
	  <td>2.5e+02</td>
	  <td>3.0</td>
	</tr>
    <tr>
	  <td>torso2</td>
	  <td>4.1e+01</td>
	  <td>1.2e+02</td>
	  <td>2.8</td>
	</tr>
    <tr>
	  <td>ecology2</td>
	  <td>9.0e+01</td>
	  <td>2.9e+02</td>
	  <td>3.2</td>
	</tr>
    <tr>
	  <td>venkat01</td>
	  <td>5.7e+01</td>
	  <td>1.2e+03</td>
	  <td>21.1</td>
	</tr>
    <tr>
	  <td>parabolic_fem</td>
	  <td>8.8e+01</td>
	  <td>3.2e+02</td>
	  <td>3.6</td>
	</tr>
    <tr>
	  <td>G3_circuit</td>
	  <td>1.2e+02</td>
	  <td>4.2e+02</td>
	  <td>3.5</td>
	</tr>
    <tr>
	  <td>memplus</td>
	  <td>4.6e+01</td>
	  <td>8.8e+03</td>
	  <td>191.8</td>
	</tr>
    <tr>
	  <td>csrp</td>
	  <td>3.8e+01</td>
	  <td>9.1e+03</td>
	  <td>2.4</td>
	</tr>
  </tbody>
</table>

<figure style="max-width: 800; margin-top: 30px;">
  <img id="fig31" src="/gh-pages/docs/docs-docsify/images/parILUT-breakdown.png" width="750" height="350" alt="parILUT" />
  <figcaption>
	Figure 3-1 Relative runtime of the building blocks in the parILUT 
	algorithm. Two bars for each problem correspond to the breakdown 
	on the two GPU architectures: Tesla V100s (left bar) and Hygon DCUs (right bar)
  </figcaption>
</figure>

### Nested Krylov Iterative Refinement Algorithm in Mixed Precision
Motivated to exploit the improved performance of lower-precision 
arithmetic, we adopt Krylov subspace methods in the conventional 
iterative refinement method to approximately solve the correction 
equation at lower precisions, denoted  
by 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>
. This so-arising nested Krylov iterative refinement (NKIR) algorithm 
allows even lower precisions 
<math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>
 at which preconditioning is applied, thus improving overall performance. 

In this section, we consider the velocity and pressure systems 
arising in lid driven cavity benchmark (
<math display="inline"><mi>R</mi><mi>e</mi><mo>=</mo><mn>100</mn></math>
) to evaluate the proposed NKIR algorithm in mixed precision. 
Contrary to Section 3.2.1, we calculate this laminar case in stationary 
and thus utilize the SIMPLE algorithm. The systems are exported from 
OpenFOAM for the evaluations shown in Tables [3-7](#tb37) and  
[3-8](#tb38). The working precision 
<math display="inline"><msub><mi>u</mi><mi>w</mi></msub></math>
 is fixed to be double precision and the relative stopping tolerance 
for the final solution and Krylov subspace method Bi-CGSTAB is 1.0e-4. 
The incomplete factor preconditioner of no fill-in is applied. Serial 
execution is performed for the NKIR method. 

The motivation is to determine the optimal precision combination at 
which the proposed mixed-precision method achieves the best overall performance. 
When applying the mixed-precision NKIR method to the velocity systems, 
several observations are made from the results in [Table 3-7](#tb37). 
By decreasing the precisions, the refinement doesn't initiate until 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><mtext>single</mtext>
</math>
 and 
<math display="inline"><msub><mi>u</mi><mi>p</mi></msub>
  <mo>=</mo><mtext>half</mtext>
</math>
, and the number of Krylov iterations is independent of precisions. Therefore, 
the best overall performance is reached at 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><mtext>single</mtext>
</math>
 and 
<math display="inline"><msub><mi>u</mi><mi>p</mi></msub>
  <mo>=</mo><mtext>half</mtext>
</math>
, which deliver an averaged speedup of 1.39x for the considered matrix 
sizes over the baseline precisions 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>double</mtext>
</math>
. Ideally, the maximum speedup would be 2.0x by considering the peak 
performance of the floating-point operations at the double and single 
precisions, see [Table 3-1](#tb31).  
Further lower precisions 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>half</mtext>
</math>
 require refinements to meet the desired relative stopping tolerance. 
This hampers the overall performance as seen from the decreased speedup. 
In summary, the optimal precision combination is 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><mtext>single, </mtext>
  <msub><mi>u</mi><mi>p</mi></msub>
  <mo>=</mo><mtext>half</mtext>
</math>
for the velocity systems.

Compared with the velocity systems, the observations for the pressure 
systems are different as seen from [Table 3-8](#tb38). The precision 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>
 at which Krylov subspace methods are applied should be kept to be 
double precision. Lower precisions lead to the unconvergence of the 
iterative refinement at the relatively large matrix size 
<math display="inline"><mi>n</mi><mo>=</mo><mn>64</mn><mo>&#xd7;</mo>
  <msup><mn>10</mn><mn>4</mn></msup>
</math>
. The optimal precision combination appears to be 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub>
  <mo>=</mo><mtext>double</mtext>
</math>
 and 
<math display="inline"><msub><mi>u</mi><mi>p</mi></msub>
  <mo>=</mo><mtext>half</mtext>
</math>
. The speedup over the total time at the baseline precisions 
<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo>
  <msub><mi>u</mi><mi>p</mi></msub>
  <mo>=</mo><mtext>double</mtext>
</math>
 is not comparable with that in [Table 3-7](#tb37).  
In the context of incompressible flows, it is usually more difficult 
to solve the pressure system than the velocity system. This leads to 
the different behavior of the mixed-precision NKIR method. 

<!-- NKIR-velocify -->
<table id="tb37">
  <caption>
	Table 3-7 Velocity systems: the number of Bi-CGSTAB iterations and iterative 
	refinements (in bracket) <br />
	with varying 
	<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>
	 and 
	<math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>
	, and the speedup over the total time at  
	<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>=
  	<math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>=
  	<math display="inline"><mtext>double</mtext></math>
	. Kunpeng ARM is utilized.
  </caption>
  <thead>
    <tr>
	  <th></th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>half</mtext></math>
	  </th>
	</tr>
	<tr>
	  <th></th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>half</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>half</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>half</mtext></math>
	  </th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td colspan="7" style="text-align: center;">Bi-CGSTAB Iter. (IR Iter.)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>3(4)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>4</mn><mo>&#xd7;</mo>
	    <msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>3(4)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>16</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>6(0)</td>
	  <td>3(4)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>64</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>5(0)</td>
	  <td>5(0)</td>
	  <td>5(0)</td>
	  <td>5(0)</td>
	  <td>5(0)</td>
	  <td>2(4)</td>
	</tr>
    <tr>
	  <td colspan="7" style="text-align: center;">Speedup</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>1.00</td>
	  <td>1.14</td>
	  <td>1.24</td>
	  <td>1.37</td>
	  <td>1.43</td>
	  <td>1.15</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>4</mn><mo>&#xd7;</mo>
	    <msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.16</td>
	  <td>1.20</td>
	  <td>1.34</td>
	  <td>1.42</td>
	  <td>1.12</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>16</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.09</td>
	  <td>1.19</td>
	  <td>1.29</td>
	  <td>1.35</td>
	  <td>1.12</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>64</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.11</td>
	  <td>1.19</td>
	  <td>1.26</td>
	  <td>1.34</td>
	  <td>1.07</td>
	</tr>
  </tbody>
</table>

<!-- NKIR pressure -->
<table id="tb38" style="margin-left: 25px;">
  <caption>
	Table 3-8 Pressure systems: the number of Bi-CGSTAB iterations and iterative 
	refinements (in bracket) <br />
	with varying 
	<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>
	 and 
	<math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>
	, and the speedup over the total time at  
	<math display="inline"><msub><mi>u</mi><mi>k</mi></msub></math>=
  	<math display="inline"><msub><mi>u</mi><mi>p</mi></msub></math>=
  	<math display="inline"><mtext>double</mtext></math>
	. Kunpeng ARM is utilized.
  </caption>
  <thead>
    <tr>
	  <th></th>
	  <th style="width: 150px;">
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th style="width: 150px;">
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th style="width: 150px;">
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th style="width: 150px;">
		<math display="inline"><msub><mi>u</mi><mi>k</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	</tr>
	<tr>
	  <th></th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>double</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>half</mtext></math>
	  </th>
	  <th>
		<math display="inline"><msub><mi>u</mi><mi>p</mi></msub><mo>=</mo><mtext>single</mtext></math>
	  </th>
	</tr>
  </thead>
  <tbody>
    <tr>
	  <td colspan="7" style="text-align: center;">Bi-CGSTAB Iter. (IR Iter.)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>80(0)</td>
	  <td>80(0)</td>
	  <td>80(0)</td>
	  <td>59(33)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>4</mn><mo>&#xd7;</mo>
	    <msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>163(0)</td>
	  <td>163(0)</td>
	  <td>165(0)</td>
	  <td>122(50)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>16</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>329(0)</td>
	  <td>329(0)</td>
	  <td>400(0)</td>
	  <td>295(100)</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>64</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>1004(0)</td>
	  <td>1005(0)</td>
	  <td>1026(0)</td>
	  <td>unconverge</td>
	</tr>
    <tr>
	  <td colspan="7" style="text-align: center;">Speedup</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><msup><mn>10</mn><mn>4</mn></msup></math></td>
	  <td>1.00</td>
	  <td>1.11</td>
	  <td>1.17</td>
	  <td>0.88</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>4</mn><mo>&#xd7;</mo>
	    <msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.10</td>
	  <td>1.14</td>
	  <td>0.77</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>16</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.11</td>
	  <td>1.01</td>
	  <td>0.83</td>
	</tr>
	<tr>
	  <td><math display="inline"><mi>n</mi><mo>=</mo><mn>64</mn><mo>&#xd7;</mo>
		<msup><mn>10</mn><mn>4</mn></msup></math>
	  </td>
	  <td>1.00</td>
	  <td>1.07</td>
	  <td>1.05</td>
	  <td>unconverge</td>
	</tr>
  </tbody>
</table>
