# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

# Clone and build hypre.

if [[ -z "$pkg_sources_dir" ]]; then
   echo "This script ($0) should not be called directly. Stop."
   return 1
fi
if [[ -z "$OUT_DIR" ]]; then
   echo "The variable 'OUT_DIR' is not set. Stop."
   return 1
fi
pkg_src_dir="hypre"
HYPRE_SOURCE_DIR="$pkg_sources_dir/$pkg_src_dir"
pkg_bld_dir="$OUT_DIR/hypre"
HYPRE_DIR="$pkg_bld_dir"
hypre_branch="${hypre_branch:-master}"
HYPRE_BRANCH="${hypre_branch}"
HYPRE_BIGINT=${hypre_big_int:+YES}
HYPRE_MIXEDINT=${hypre_mixed_int:+YES}
HYPRE_DEBUG=${hypre_debug:+YES}
pkg_var_prefix="hypre_"
pkg="hypre"


function hypre_clone()
{
   pkg_repo_list=("git@github.com:hypre-space/hypre.git"
                  "https://github.com/hypre-space/hypre.git")
   pkg_git_branch="$hypre_branch"
   cd "$pkg_sources_dir" || return 1
   if [[ -d "$pkg_src_dir" ]]; then
      update_git_package
      return
   fi
   for pkg_repo in "${pkg_repo_list[@]}"; do
      echo "Cloning $pkg from $pkg_repo ..."
      git clone "$pkg_repo" "$pkg_src_dir" && return 0
   done
   echo "Could not successfully clone $pkg. Stop."
   return 1
}


function hypre_build()
{
   if package_build_is_good; then
      echo "Using successfully built $pkg from OUT_DIR."
      return 0
   elif [[ ! -d "$pkg_bld_dir" ]]; then
      echo "Cloning hypre from $HYPRE_SOURCE_DIR to OUT_DIR ..."
      cd "$OUT_DIR" && git clone "$HYPRE_SOURCE_DIR" || {
         echo "Cloning $HYPRE_SOURCE_DIR to OUT_DIR failed. Stop."
         return 1
      }
   fi
   local my_cflags="$CFLAGS"
   if [ -n "$hypre_debug" ]; then
      my_cflags="-g"
   fi
   echo "Building $pkg, sending output to ${pkg_bld_dir}_build.log ..." && {
      cd "$pkg_bld_dir/src" && \
      if [[ -e config/Makefile.config ]]; then
         make distclean
      fi && \
      ./configure \
         CC="$MPICC" \
         CXX="$MPICXX" \
         CFLAGS="$my_cflags" \
         CXXFLAGS="$my_cflags" \
         $HYPRE_EXTRA_CONFIG \
         ${hypre_big_int:+--enable-bigint} \
         ${hypre_mixed_int:+--enable-mixedint} \
         ${hypre_debug:+--enable-debug} \
         --disable-fortran \
         --without-fei && \
      make -j $num_proc_build
   } &> "${pkg_bld_dir}_build.log" || {
      echo " ... building $pkg FAILED, see log for details."
      return 1
   }
   echo "Build successful."
   print_variables "$pkg_var_prefix" \
      HYPRE_BRANCH HYPRE_BIGINT HYPRE_MIXEDINT HYPRE_DEBUG \
      > "${pkg_bld_dir}_build_successful"
}


function build_package()
{
   hypre_clone && get_package_git_version && hypre_build
}
