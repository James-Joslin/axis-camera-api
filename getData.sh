# To be run in the directory where you want the data

fileName=CrowdHuman_train01.zip
fileId=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

fileName=CrowdHuman_train02.zip
fileId=17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

fileName=CrowdHuman_train03.zip
fileId=1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

fileName=CrowdHuman_val.zip
fileId=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
   
fileName=annotation_train.odgt
fileId=1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
   
fileName=annotation_val.odgt
fileId=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

unzip CrowdHuman_train01 -d ./CrowdHuman_train01
unzip CrowdHuman_train02 -d ./CrowdHuman_train02
unzip CrowdHuman_train03 -d ./CrowdHuman_train03
unzip CrowdHuman_val.zip -d ./CrowdHuman_val

rm CrowdHuman_train01.zip 
rm CrowdHuman_train02.zip 
rm CrowdHuman_train03.zip 
rm CrowdHuman_val.zip

rm /tmp/cookie