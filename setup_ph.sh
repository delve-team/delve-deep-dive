
DIPHA_PATH=$(pwd)/dipha/dipha
PERSEUS_PATH=$(pwd)/Deps/perseus

git submodule update --init --recursive

cd dipha
cmake CMakeLists.txt && make

cd ..
mkdir Deps
cd Deps
wget http://people.maths.ox.ac.uk/nanda/source/perseus_4_beta.zip
unzip -u perseus_4_beta.zip
g++ Pers.cpp -O3 -fpermissive -o perseus

cd ..


sed -i "s#dipha=#dipha=$DIPHA_PATH#" tda-toolkit/pershombox/_software_backends/software_backends.cfg
sed -i "s#perseus=#perseus=$PERSEUS_PATH#" tda-toolkit/pershombox/_software_backends/software_backends.cfg
