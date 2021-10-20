# create the conda environment "ml4p"
echo ""
conda env create --file conda/ml4p.yml

# # check for latex, needed for atlas_mpl_style
# if ! command -v latex &> /dev/null
# then
#     echo "Did not find latex. Installing tex-live 2020."
#     tar -xf tex/install-tl-unx.tar.gz
#     ./install-tl-20201021/install-tl --profile=tex/texlive.profile
#     export PATH="/usr/local/texlive/2020/bin/x86_64-linux:$PATH"
#     echo "Finished installing tex-live 2020"
#     rm -r ./install-tl-20201021
#     exit
# fi


