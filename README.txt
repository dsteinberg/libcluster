Libcluster - README file

Author: Daniel Steinberg
        Australian Centre for Field Robotics
        The University of Sydney

Date:   12/04/2012

This library implements the following algorithms, classes and functions:

    - The Variational Dirichlet Process (VDP) [1]

    - The Bayesian Gaussian Mixture Model [2]

    - The Grouped Mixtures Clustering (GMC) model [3]

    - The Symmetric Grouped Mixtures Clustering (S-GMC) model [3]
    
    - And more clustering algorithms based on diagonal Gaussian, and 
      Exponential distributions.

    - Various functions for evaluating means, standard deviations, covariance,
      primary eigenvalues etc of data.

    - Various tools, including AUV post-processing pipeline tools.

All algorithms can be run in an incremental fashion.

 [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
     Dirichlet process mixtures, Advances in Neural Information Processing
     Systems, vol. 19, p. 761, 2007.

 [2] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
     Springer Science+Business Media, 2006.
     
 [3] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Clustering Groups of
     Related Visual Datasets," unpublished, 2011.


DEPENDENCIES -------------------------------------------------------------------

- Eigen version 3
- Boost version 1.4.x or greater
- For building the AUV pipeline tools (optional):
  + libplankton
  + auv_data_tools
  + seabed_slam_file_io.hpp


INSTALL INSTRUCTIONS (Linux/OS X) ----------------------------------------------

To build libcluster:

1) Make sure you have CMake installed, and Eigen and Boost preferably in the 
   usual locations:

        /usr/local/include/eigen3/ or /usr/include/eigen3
        /usr/local/include/boost or /usr/include/boost

  (These are in the repositories as of Ubuntu 11.10)

2) Make sure you have checked out CMakeUtils to the same location as libcluster

3) Make a build directory where you checked out the source if it does not
   already exist, then change into this directory,

    cd {where you checked out the source}
    mkdir build
    cd build

4) To build libcluster INCLUDING the AUV pipeline tools, run the following from
    the build directory:

        cmake ..
        make
        sudo make install

   This installs:
        libcluster.h    /usr/local/include
        probutils.h     /usr/local/include
        libcluster.*    /usr/local/lib          (* this is either .dylib or .so)
        vdp_label       /usr/local/bin       (config options in vdp_cluster.cfg)
        
   To build without the AUV pipeline tools, see the notes below.

5) Use the doxyfile in {where you checked out the source}/doc to make the
   documentation with doxygen:

        doxygen Doxyfile

NOTE:
 - There are few options you can change using ccmake (or the cmake gui), these 
   include:
   
   BUILD_GREEDY_SPLIT (toggle ON or OFF, default OFF)
      This uses the greedy cluster split heuristic instead of the exhaustive
      heuristic. It is MUCH faster, but does give different results. I have yet
      to determine whether this is actually worse than the exhaustive method.
      
   BUILD_PIPELINE_TOOLS (toggle ON or OFF, default ON)
      This builds the AUV pipeline tools - which at this stage is basically just
      an interface for the VDP clustering algorithm. This has EXTRA 
      dependencies.
      
   CMAKE_INSTALL_PREFIX (default /usr/local)
      The default prefix for installing the library and binaries.
      
   EIGEN_INCLUDE_DIRS (default /usr/include/eigen3)
      Where to look for the Eigen matrix library.  
    
 - On linux you may have to run "sudo ldconfig" before the system can find
   libcluster.so.


MATLAB INTERFACE (Linux/OS X) --------------------------------------------------

I have included a mex interface for using this library with Matlab. You just
need to make sure that:

  a) You have used a 32 bit compiler if you have 32 bit Matlab (or 64 bit 
     compiler for 64 bit Matlab).

  b) The compiler you have used is similar to Matlab's (I have found that if you
      are off by a minor version number it is ok still).

To build the Matlab interface:

  1) In Matlab, navigate to {where you checked out the source}/matlab.

  2) If Eigen is installed at 

        /usr/include/eigen3

     type MakeFile.m and enter. Otherwise specify the location of Eigen in 
     MakeFile.m and execute.

  3) If compilation succeeds, you will be prompted if you want to copy the
     binaries and .m files to a location of your choice, and update Matlab's 
     path. If you answer no, you will have to do this manually.

  4) Done! Have a look at the .m files for documentation on how to use the
     interface.

Notes:

 - I have included the scripts SS2GMM.m and SS2EMM.m to turn the Sufficient 
   Statistics structure (SS) into Gaussian mixture model and Exponential mixture
   model structure respectively.

 - Either the build will warn you, or running the .m files will fail if your
   compiler is not compatible with Matlab. To fix this with Ubuntu 10.10 and
   11.04 I did the following:

   1) run $ sudo apt-get install gcc-4.x-base g++-4.x-base libstdc++6-4.x-dev
      Where 'x' is the version number that Matlab uses (or close too 4.3 for
      10.10 or 4.4 for 11.04 seems to work).

   2) Built the library as in the install instructions, BUT replaced:

      cmake ..

      with

      CC=gcc-4.x CXX=g++-4.x cmake ..

   3) Built the Matlab interface using the above instructions, but changed the
      'compiler' string in MakeFile.m to 'CXX=g++-4.x CC=g++-4.x LD=g++-4.x'

 - If you are issued with a warning something along the lines of:

    ??? Invalid MEX-file '.../vdpcluster_mex.mexa64':
    /../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.11' not found
    (required by /.../libcluster.so).

    try one of the following:

    a) change the symbolic link ~/.maltab/bin/gcc to point to the version of gcc
       you are using in /usr/bin/gcc-4.x

    b) if no such directory exists you can remove the symlinks in Matlab's root 
       to its own copy's of the C and C++ libraries, so it will then use your 
       systems,

        $ cd <MATLABROOT>/sys/os/glnxa64
        $ mv libstdc++.so.6.0.10 libstdc++.so.6.0.10.bak  (or closest version)
        $ mv libstdc++.so.6 libstdc++.so.6.bak
        $ mv libgcc_s.so.1  libgcc_s.so.1.bak

       Matlab should now automatically look for the correct system libraries.

- If you are using OS X, and the version of matlab you are using cannot find the
  system heaters, have a look at ~/.matlab/R20xxx/mexopts.sh and change all 
  lines:

	    SDKROOT='/Developer/SDKs/MacOSX10.X.sdk'
      MACOSX_DEPLOYMENT_TARGET='10.X'

  To your version of OS X, e.g. 10.7.


USABILITY TIPS -----------------------------------------------------------------

When verbose mode is activated you will get output that looks something like
this:

   Learning MODEL X...
   --------<=>
   ---<==>
   --------x<=>
   --------------<====>
   Finished!
   Number of clusters = 4
   Free Energy = 41225

What this means:
   '-' iteration of Variational Bayes (VBE and VBM step)
   '<' cluster splitting has started (model selection)
   '=' found a valid candidate split
   '>' chosen candidate split and testing for inclusion into model
   'x' clusters have been deleted because they became devoid of observations

For best clustering results, I have found the following tips may help:

0)  If clustering runs REALLY slowly then it may be because of hyper-threading.
    OpenMP will by default use as many cores available to it as possible, this 
    includes virtual hyper-threading cores. Unfortunately this nearly always
    results in large slow-downs, so try only allowing these functions to use
    a number of threads less than or equal to the number of PHYSICAL cores on 
    your machine.

1)  Garbage in = garbage out. Make sure your assumptions about the data are 
    reasonable for the type of cluster distribution you use. For instance, if  
    your observations do not resemble a mixture of Gaussians in feature space,
    then it may not be appropriate to use Gaussian clusters.

2)  For Gaussian clusters: standardising or whitening your data may help, i.e.

    if X is an NxD matrix of observations you wish to cluster, you may get
    better results if you use a standardised version of it, X*,

      X_s = C * ( X - mean(X) ) / std(X)

    where C is some constant (optional) and the mean and std are for each 
    column of X.
    
    You may obtain even better results by using PCA or ZCA whitening on X:
    
      [U, S, V] = svd(cov(X));
    
      X_w = X * U * diag(1./sqrt(diag(S))) * U';   % ZCA Whitening
      
    Such that cov(X_w) = I_D.
    
    Also, to get some automatic scaling you can multiply the prior by the 
    PRINCIPAL eigenvector of cov(X) (or cov(X_s), cov(X_w)).
    
    NOTE: If you use diagonal covariance Gaussians I STRONGLY recommend PCA or 
          ZCA whitening your data first, otherwise you may end up with hundreds
          of clusters!
          
3)  For Exponential clusters: Your observations have to be in the range [0,inf).
    The clustering solution may also be sensitive to the prior. I find usually 
    using a prior value that has the approximate magnitude of your data or more
    leads to better convergence.


COMMAND LINE INTERFACES --------------------------------------------------------

WARNING: This is only a really rough and ready version.

The command line interface is:

    $ vdp_cluster [feature file] [cluster result file]

    e.g. try

    $ vdp_cluster {source}/test/scott25.data  cresults.dat

So far this just reads text files of the type in {source}/test and outputs other
text files.
