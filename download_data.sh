if [ ! -d data/ ]; then
    mkdir -p data/
fi

cd data

curl -L https://www.dropbox.com/s/3hoc0qmeov8ro0r/scannet.zip?dl=1 > scannet.zip

unzip scannet.zip
rm scannet.zip

cd ..

curl -L https://www.dropbox.com/s/umcyvto9nomhjt2/mesh_out.zip?dl=1 > mesh_out.zip

unzip mesh_out.zip
rm mesh_out.zip
