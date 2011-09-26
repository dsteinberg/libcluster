% Run this file to make the matlab interface for the libgbnp clustering
% algorithms
%
% NOTE: you need to compile the c++ source first using cmake.

%% Change these variables for your setup.

% Default install path
instpath = strcat(strrep(userpath, ':', ''), '/libcluster/');

% Compiler options
eigendir  = '/usr/local/include/eigen3';
include   = '../include';
clusterdylib = '../lib';
compiler  = 'CXX=g++ CC=g++ LD=g++';

%% Compile... Think twice about changing stuff in here.

% Get dynamic library name
dlname = dir(strcat(clusterdylib,'/libcluster.*'));
clusterdylib = strcat(clusterdylib,'/',dlname.name);

% Create compile string for vdpcluster
vdpcluster_cmp = sprintf( ...
    'mex %s -I%s -I%s vdpcluster_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Create compile string for gmccluster
gmccluster_cmp = sprintf( ...
    'mex %s -I%s -I%s gmccluster_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Create compile string for igmccluster
igmccluster_cmp = sprintf( ...
    'mex %s -I%s -I%s igmccluster_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Create compile string for igmcclassify
igmcclassify_cmp = sprintf( ...
    'mex %s -I%s -I%s igmcclassify_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Create compile string for gmmclassify
gmmclassify_cmp = sprintf( ...
    'mex %s -I%s -I%s gmmclassify_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Create compile string for gmmpredict
gmmpredict_cmp = sprintf( ...
    'mex %s -I%s -I%s gmmpredict_mex.cpp intfctns.cpp %s', ...
    compiler, eigendir, include, clusterdylib ...
    );

% Compile!
fprintf('Compiling vdpcluster...\n')
eval(vdpcluster_cmp);
fprintf('Compiling gmccluster...\n')
eval(gmccluster_cmp);
fprintf('Compiling igmccluster...\n')
eval(igmccluster_cmp);
fprintf('Compiling igmcclassify...\n')
eval(igmcclassify_cmp);
fprintf('Compiling gmmclassify...\n')
eval(gmmclassify_cmp);
fprintf('Compiling gmmpredict...\n')
eval(gmmpredict_cmp);
fprintf('Finished!\n')

%% Install and add to path

ansstr = input('Install files and add to Matlab path? y/n [n]:', 's');

if strcmp(ansstr,'y') || strcmp(ansstr,'Y'),
    instsr = input(sprintf('Where would you like to install the files? [%s]:', instpath),'s');
    if ~isempty(instsr), instpath = instsr; end
    
    eval(sprintf('!mkdir %s', instpath));
    
    eval(sprintf('!cp vdpcluster_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp vdpcluster.m %s', instpath));
        
    eval(sprintf('!cp gmccluster_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp gmccluster.m %s', instpath));
    
    eval(sprintf('!cp igmccluster_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp igmccluster.m %s', instpath));
    eval(sprintf('!cp createigmc.m %s', instpath));
    eval(sprintf('!cp setcounts.m %s', instpath));
    
    eval(sprintf('!cp igmcclassify_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp igmcclassify.m %s', instpath));
    
    eval(sprintf('!cp gmmclassify_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp gmmclassify.m %s', instpath));
    
    eval(sprintf('!cp gmmpredict_mex.%s %s', mexext, instpath));
    eval(sprintf('!cp gmmpredict.m %s', instpath));
    
    eval(sprintf('!cp convgmma2c.m %s', instpath));
    eval(sprintf('!cp convgmmc2a.m %s', instpath));
    
    addpath(instpath);
    savepath
end
