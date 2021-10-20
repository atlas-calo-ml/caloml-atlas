

FILE=~/.local/share/fonts/Humor-Sans.ttf
if test -f "$FILE"; then
    echo "$FILE exists."
else
    # Get the Humor Sans font -- for xkcd-style plots in matplotlib
    wget https://github.com/shreyankg/xkcd-desktop/blob/master/Humor-Sans.ttf
    # move this file to a local font location (for the user, not system)
    mkdir -p ~/.local/share/fonts
    mv Humor-Sans.ttf ~/.local/share/fonts/
fi

# Activating conda env
# For the following line, see https://stackoverflow.com/questions/2559076/how-do-i-redirect-output-to-a-variable-in-shell
read conda_str1 < <(conda info | grep -i 'base environment' | sed -e 's/[ ]*(.*$//g' | sed -e 's/[ ]*base environment :[ ]*//g')
conda_str2="/etc/profile.d/conda.sh"
conda_str="$conda_str1$conda_str2"
source "$conda_str"
conda activate ml4p

# rebuild the mpl font cache
python setup.py