## Model 1
Epoch = 120
Weight Decay = 0.001 
LR = 0.001
Step Size = 500
Validation step = 25 
Train ROI Images = 100
Layers = heads
Write any other changes here = None
Better Model yet 

## Model 2
Epoch = 60
Weight Decay = 0.001 
LR = 0.01
Step Size = 1000
Validation step = 50 
Train ROI Images = 330
Layers = heads
Write any other changes here = none 

## Model 3
## Worst, rejected model
Epoch = 120
Weight Decay = 0.01
LR = 0.001, 0.0001, 1e-5
Step Size = 500
Validation step = 25
Train ROI Images = 330
Layers = heads
Write any other changes here = none


## Model 4 
## not working; REJECTED
Epoch = 30
Weight Decay = 0.001
LR = 0.001, 0.0001
Step Size = 500
Validation step = 25
Train ROI Images = 330
Layers = heads
Write any other changes here = nothing
```
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 0.3,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
```

## Model 6
## Results
At about 350th it started giving NAN, but at 232nd epoch it improved and is giving good predictions(a few) and alot of bad predictions.
Epoch = 2000
Weight Decay = 0.001
LR = 0.001, 0.0001
Step Size = 100
Validation step = 5
Train ROI Images = 100
Layers = heads
Write any other changes here = RPN_NMS_THRESHOLD back to 0.7 

## Model 7
At about 300+ epoch NaN, and it did not improve after 198th epoch
Epoch = 447/2000
Weight Decay = 0.001
LR = 0.001
Step Size = 100
Validation step = 5
Train ROI Images = 100
Layers = all
Write any other changes here = postive roi ration previously 50% now 70% 

## Model 8
## I noticed that thorugh out the training the val loss and overall loss kept improving but the results are not that good at all. Very bad results. Took me almost 17 hours to train. I did not save weights. 
Epoch = 120
Weight Decay = 0.001 
LR = 0.001, 0.0001
Step Size = 500
Validation step = 250 
Train ROI Images = 250
Layers = all
Write any other changes here = 50% roi ratio 

## Model 9
Epoch = 120
Weight Decay = 0.001 
LR = 0.001, 0.0001
Step Size = 500
Validation step = 50
Train ROI Images = 200
Layers = heads
Write any other changes here = nothing


## Model 10
## This failed, maybe after 60 or 80 epoch or maybe 40 epoch I shouldd've lowered the LR
Epoch = 240
Weight Decay = 0.001 
LR = 0.001, 0.0001
Step Size = 500
Validation step = 50 
Train ROI Images = 200
Layers = heads
Write any other changes here = 


## Model 11
Epoch = 
Weight Decay = 
LR = 
Step Size = 
Validation step = 
Train ROI Images = 
Layers = 
Write any other changes here = 


## Model 12
Epoch = 
Weight Decay = 
LR = 
Step Size = 
Validation step = 
Train ROI Images = 
Layers = 
Write any other changes here = 


## Model 13
Epoch = 
Weight Decay = 
LR = 
Step Size = 
Validation step = 
Train ROI Images = 
Layers = 
Write any other changes here = 
