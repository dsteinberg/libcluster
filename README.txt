Grouped Bayesian and Non-Parametric clustering Library - libgbnp - README file

Author: Daniel Steinberg
        Australian Centre for Field Robotics
        The University of Sydney

Date:   30/05/2011

This library implements the following algorithms, classes and functions:

    - The Variational Dirichlet Process (VDP) [1]
    - The Grouped Mixtures Clustering (GMC) model [2]
    - Gaussian mixture model class that can be learned by these two algorithms
      and can be used for classifying new data, and also density
      estimation/prediction.
    - Various functions for evaluating means, covariances, primary eigenvalues
      etc of data.


 [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
     Dirichlet process mixtures, Advances in Neural Information Processing
     Systems, vol. 19, p. 761, 2007.

 [2] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Hierarchal Bayesian
     mixtures for clustering multiple related datasets." NIPS 2011
     Submission, June 2011.


INSTALL INSTRUCTIONS (Linux/OS X) ----------------------------------------------

To build libgbnp:

1) Make sure you have CMake (2.6 +), Boost (1.4.x) and Eigen 3 installed.
   Preferably in the usual locations:

        /usr/local/inlcude/eigen3/
                                       (As of ubuntu 11.04, apt-get only has v2)
        /usr/local/include/boost or /usr/include/boost
                                                    (Just use apt-get on ubuntu)

2) Make a build directory where you checked out the source if it does not
   already exist, then change into this directory,

    cd {where you checked out the source}
    mkdir build
    cd build

3) Run the following from the build directory:

        cmake ..
        make
        sudo make install

   This installs:
        libgbnp.h       /usr/local/include
        gbnputils.h     /usr/local/include
        libgbnp.*       /usr/local/lib      (* this is either .dylib or .so)
        vdpcluster      /usr/local/bin

4) Use the doxyfile in {where you checked out the source}/doc to make the
   documentation with doxygen.

Notes:
 - On ubuntu 10.10 I had to run "sudo ldconfig" before the system could find
   libgbnp.so.


MATLAB INTERFACE (Linux/OS X) --------------------------------------------------

I have included a mex interface for using this library with Matlab. You just
need to make sure that:
a) You have used a 32 bit compiler if you have 32 bit Matlab etc.
b) The compiler you have used is similar to Matlab's (I have found that if you
    are off by a minor version number it is ok still).

To build the Matlab interface:

1) In Matlab, navigate to {where you checked out the source}/matlab.

2) Type MakeFile.m and enter.

3) If compilation succeeds, you will be prompted if you want to copy the
   binaries and .m files to a location of your choice, and update Matlab's path.
   If you answer no, you will have to do this manually.

4) Done! Have a look at the .m files for documentation on how to use the
   interface.

Notes:
 - If you get an error saying the Eigen libraries can't be found, then you can
   change the path to Eigen in MakeFile.m.

 - Either the build will warn you, or running the .m files will fail if your
   compiler is not compatible with Matlab. To fix this with Ubuntu 10.10 and
   11.04 I did the following:

   1) run $ sudo apt-get install gcc-4.x-base g++-4.x-base libstdc++6-4.x-dev
      Where 'x' is the version number that Matlab uses (or close too 4.3 for
      110.10 or 4.4 for 1.04 seem to work).

   2) Built the library as in the install instructions, BUT replaced:

      cmake ..

      with

      CC=gcc-4.x CXX=g++-4.x cmake ..

   3) Built the Matlab interface using the above instructions, but changed the
      'compiler' string in MakeFile.m to 'CXX=g++-4.x CC=g++-4.x LD=g++-4.x'

      If the above still fails (like it did for me on Ubuntu 11.04), change the
      symbolic link ~/.maltab/bin/gcc to point to the version of gcc you are
      using in /usr/bin/gcc-4.x

 - If you are issued with a warning something along the lines of:

    ??? Invalid MEX-file '.../vdpcluster_mex.mexa64':
    /../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.11' not found
    (required by /.../libgbnp.so).

   you can remove the symlinks in Matlab's root to its own copy's of the C and
   C++ libraries, so it will then use your systems,

    $ cd <MATLABROOT>/sys/os/glnxa64
    $ mv libstdc++.so.6.0.10 libstdc++.so.6.0.10.bak   (or your closest version)
    $ mv libstdc++.so.6 libstdc++.so.6.bak
    $ mv libgcc_s.so.1  libgcc_s.so.1.bak

   Matlab should now automatically look for the correct system libraries.


USABILITY TIPS -----------------------------------------------------------------

For best results, I have found the following tips may help:

1)  Look at a histogram of each dimension of your data if possible. Make sure it
    looks somewhat like a mixture of Gaussians.

2)  Pre-scaling your data may help too. For instance, when you have all of your
    data, standardising it will help, i.e.

    if X is an NxD matrix of observations you wish to cluster, you may get
    better results if you use a standardised version of it, X*,

    X* = C * ( X - mean(X) ) / std(X)

    where C is some constant (usually I use 10).

3)  If you get "calculated a negative free energy" error, make sure you data is
    scaled well. Sometimes this error can be caused by the total variance of you
    data being small enough that the resulting Gaussian likelihoods can be
    greater than one, causing a negaive free energy. This may not be a problem
    -- however I explicitly check for this condition.


COMMAND LINE INTERFACES --------------------------------------------------------

TODO.

The command line interface is:

    $ vdpcluster [feature file] [gmm result file] [cluster result file]

    e.g. try

    $ vdpcluster {where you checked out the source}/test/scott25.dat gmm.dat
        cresults.dat

So far this just reads text files of the type in {where you checked out the
source}/test and outputs other text files. I haven't invested too much effort
in, since we need to still figure out a way to store the feature files.


NOTES/TODO ---------------------------------------------------------------------

1) Finalise the command line interfaces for the VDP and GMC.
2) Incremental versions coming your way soon (I hope).