%comprobar el tamaño de las imagenes del dataset

rootfolder='DATASETS\DATASET_CNN';
categories={'Car','Motorbike','Face'};
imds = imageDatastore(fullfile(rootfolder, categories),'Labelsource','foldernames');
imds.ReadSize = numpartitions(imds)
imds.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
for i=17000:length(imds.Files)
    img1=readimage(imds,i-1);
    img2=readimage(imds,i);
    if size(img1) ~= size(img2)
        size(img1)
        size(img2)
        i
    end
end
        