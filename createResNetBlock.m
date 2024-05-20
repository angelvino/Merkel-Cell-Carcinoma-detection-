% Define a function to create a ResNet block
function layers = createResNetBlock(numFilters, filterSize, block, name, stride)
    layers = [
        convolution2dLayer(1, numFilters, 'Stride', stride, 'Name', ['res',block,'_',name,'_branch2a'])
        batchNormalizationLayer('Name', ['bn',block,'_',name,'_branch2a'])
        reluLayer('Name', ['relu',block,'_',name,'_branch2a'])
        
        convolution2dLayer(filterSize(1), numFilters, 'Padding', 'same', 'Name', ['res',block,'_',name,'_branch2b'])
        batchNormalizationLayer('Name', ['bn',block,'_',name,'_branch2b'])
        reluLayer('Name', ['relu',block,'_',name,'_branch2b'])
        
        convolution2dLayer(1, 4 * numFilters, 'Name', ['res',block,'_',name,'_branch2c'])
        batchNormalizationLayer('Name', ['bn',block,'_',name,'_branch2c'])
        
        additionLayer(2, 'Name', ['res',block,'_',name,'_add'])
        reluLayer('Name', ['relu',block,'_',name,'_out'])
    ];
end