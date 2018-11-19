// ARFF-App
const fs = require('fs');
require('@tensorflow/tfjs-node')
global.fetch = require('node-fetch')
const tf = require('@tensorflow/tfjs')
const nifti =  require('nifti-reader-js')


function imgBytes2Values(bytesArr, bitsPerVoxel) {
    var niftiImage;
    if(bitsPerVoxel == 16) {
        niftiImage = Array.from(new Uint16Array(bytesArr))
    } else if(bitsPerVoxel == 32)
        niftiImage = Array.from(new Uint32Array(bytesArr))
    else if(bitsPerVoxel == 64) {  
        niftiImage = Array.from(new Float64Array(bytesArr))
    } else {console.log('Unsual number of bits per voxel.')}
    return(niftiImage)
}

function getNiftiShape(niftiHeader) {
    dims = niftiHeader['dims']
    shape = [dims[1], dims[2], dims[3]]
    return(shape);
}

function getNiftiTensor(fp) {
    // Get nifti file ArrayBuffer
    const data = fs.readFileSync(fp)
    var buf = data.buffer
    // Check if data compressed
    if(nifti.isCompressed(buf)) {
        buf = nifti.decompress(buf)
    }
    // Check if data is in NIFTI format per package
    if(nifti.isNIFTI(buf)) {
        // Nifti Header
        const niftiHeader = nifti.readHeader(buf)
        //console.log(niftiHeader.toFormattedString())
        // Nifti Image
        const niftiImageRaw = nifti.readImage(niftiHeader, buf)
        bitsPerVoxel = niftiHeader['numBitsPerVoxel']
        const niftiImage = imgBytes2Values(niftiImageRaw, bitsPerVoxel)
        const shape = getNiftiShape(niftiHeader)
        niftiTensor = tf.tensor3d(niftiImage, shape)
    }
    return niftiTensor;
}

function resizePadding(niftiTensor, outShape, value) {
    // Orig. shape
    var w = niftiTensor.shape[0],
        h = niftiTensor.shape[1],
        d = niftiTensor.shape[2];
    // Desired Shape
    var outW = outShape[0],
        outH = outShape[1],
        outD = outShape[2];
    // Delta b.w. desired and orig.
    var deltaW = outW - w,
        deltaH = outH - h,
        deltaD = outD - d;
    // Construct paddings -  @TODO  what if delta odd?
    paddings = [[deltaW/2, deltaW/2], [deltaH/2, deltaH/2], [deltaD/2, deltaD/2]]
    // Pad Tensor
    paddedTensor = tf.pad(niftiTensor, paddings=paddings, constantValue=value)
    return paddedTensor;
}

function resizeNiftiTensor(niftiTensor, outShape, mask = true) {
    // Resize w. different methods based on image or mask
    var resizedTensor;
    if(mask) {
        resizedTensor = resizePadding(niftiTensor, outShape=outShape, value=1)
    } else {
        resizedTensor = resizePadding(niftiTensor, outShape=outShape, value=0)
    }
    return resizedTensor;
}

function loadModelLocal(model_json_fp) {
    // Load the model
    var model;
    if(fs.existsSync(model_json_fp)) {
        const localPath = `file://${path}`
        var model = tf.loadModel(local_path);
    } else {
        console.log('Model json could not be found.')
    }
    return model;
}

function loadModelRemote(path) {
    const model = tf.loadModel(path)
    return model;
}

console.log('H4xoR i7!')
// Load & Preprocess Data
const img_path = '/Users/mechaman/Documents/ARFF-App/data/ixi/IXI002-Guys-0828-T1.nii.gz'
var inputTensor = getNiftiTensor(img_path)
var outShape = [256, 320, 256]
console.log(inputTensor.shape)
var resizedInputTensor = resizeNiftiTensor(inputTensor, outShape, mask=true)
console.log(resizedInputTensor.shape)
// Load Model
const model_path = 'http://127.0.0.1:8080/model.json'
var model = loadModelRemote(model_path)
console.log('Model Loaded.')
// Generate Binary Mask
//const binary_mask = model.predict(resizedInputTensor)
//console.log('Generated binary mask of size : ' + binary_mask)