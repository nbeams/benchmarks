# This file is part of CEED. For more details, see exascaleproject.org.

if [[ -z "$root_dir" ]]; then
   echo "This script ($0) should not be called directly. Stop."
   return 1
fi

source "$test_dir/dtp.sh.inl"


function build_and_run_tests()
{
    setup_occa_test

    # Run the tests
    local test_name=dtp_occa
    $dry_run cd "$test_exe_dir"
    for i in "${!case_orders[@]}"; do
        $dry_run ./$test_name -d "mode:'CUDA',deviceID:0" --order "${case_orders[$i]}" --ref-levels "${case_refs[$i]}" -ac --preconditioner none --mesh "fichera.mesh"
    done
}


test_required_packages="metis hypre occa acrotensor mfem-occa"
