## What is in the folder

This folder contains the source code for running a common solution routine for the optimal solution of the climate-economy models. The solution results as well as pretrained models for a replication are stored in the folder [optimal_results](optimal_results).

## How to run the optimal solution routine
To start the computation from scratch, change the following specifications in the config file (config/config.yaml),
while leaving the other entries untouched:

```
defaults:
  - constants: dice_generic_XXX
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables:  dice_generic_XXX


MODEL_NAME:  dice_generic
```
XXX stands for the specific parametrisation of the model, that is presented below.

Thereafter, make sure you are at the root directory of DEQN (e.g., ~/DEQN_for_IAMs), and
execute:

```
$ python run_deepnet.py
```
## How to analyze the pre-computed solutions

To analyze the the raw results used for the figures presented in the paper, you need to perform two steps.

```
$ export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Climate_in_Climate_Economics/DEQN_for_IAMS/<MODEL_FOLDER>

$ python post_process_generic.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR

```

For more details regarding the postprocessing of results, please go to the README [here](../README.md).

## How can I replicate the same training routine for the neural network?

For some models the neural network cannot be trained from scratch. To solve these models one needs to rely on the pretrained neural network which can be utilized for further training to solve the problem at hand. For each model below it is indicated how to replicate the solution if pretraining is needed. In case pretraining is needed, one should take the solution of another model specified in the instructions as a pretrained neural net. The training process of the pretrained neural net should be restarted from the latest checkpoint with the parameters of the problem at hand. When nothing is mentioned then the model can be solved from scratch.

## Which models can I run with this routine?
This routine can be used to find an optimal solution to the following models:

**CDICE:**

To run the model:

```
  - constants: dice_generic_mmm_mmm
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_mmm
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/cdice
```

********************************************************************************

**CDICE-GISS-E2-R:**

To run the model:

```
  - constants: dice_generic_mmm_giss
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_giss
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/cdice_giss
```

********************************************************************************

**CDICE-HadGEM2-ES:**

To run the model:

```
  - constants: dice_generic_mmm_hadgem
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_hadgem
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/cdice_hadgem
```

********************************************************************************

**CDICE-LOVECLIM:**

To run the model:

```
  - constants: dice_generic_mmm_loveclim
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_loveclim
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/cdice_loveclim
```

********************************************************************************

**CDICE-MESMO:**

To run the model:

```
  - constants: dice_generic_mmm_mesmo
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_mesmo
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/cdice_mesmo
```
To replicate the neural network training:
Use the solution to the CDICE model as a pretrained neural network

********************************************************************************

**DICE-2016:**

To run the model:

```
  - constants: dice_generic_2016
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/dice2016
```

********************************************************************************

**DICE-2016, ECS=2.15:**

To run the model:

```
  - constants: dice_generic_2016_ecs215
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/dice2016_ecs215
```

To replicate the neural network training:
Use the solution to the DICE-2016 model as a pretrained neural network

********************************************************************************

**DICE-2016, ECS=4.55:**

To run the model:

```
  - constants: dice_generic_2016_ecs455
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/dice2016_ecs455
```
To replicate the neural network training:
Use the solution to the DICE-2016 model as a pretrained neural network

********************************************************************************

**CDICE-LOVECLIM-GISS-E2-R:**

To run the model:

```
  - constants: dice_generic_loveclim_giss
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_loveclim_giss
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/loveclim_giss
```
To replicate the neural network training:
Use the solution to the CDICE-GISS-E2-R model as a pretrained neural network


********************************************************************************

**CDICE-LOVECLIM-HadGEM2-ES:**

To run the model:

```
  - constants: dice_generic_loveclim_hadgem
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_loveclim_hadgem
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/loveclim_hadgem
```

To replicate the neural network training:
Use the solution to the CDICE-HadGEM2-ES model as a pretrained neural network

********************************************************************************

**CDICE-MESMO-GISS-E2-R:**

To run the model:

```
  - constants: dice_generic_mesmo_giss
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mesmo_giss
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/mesmo_giss
```

To replicate the neural network training:
Use the solution to the CDICE-GISS-E2-R model as a pretrained neural network

********************************************************************************

**CDICE-MESMO-HadGEM2-ES:**

To run the model:

```
  - constants: dice_generic_mesmo_hadgem
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mesmo_hadgem
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/mesmo_hadgem
```
To replicate the neural network training:
Use the solution to the CDICE-HadGEM2-ES model as a pretrained neural network

********************************************************************************

**DICE-2016, psi=2.:**

To run the model:

```
  - constants: dice_generic_2016_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_dice16_psi2
```

********************************************************************************

**DICE-2016, psi=2., ECS=2.15:**

To run the model:

```
  - constants: dice_generic_2016_ecs215_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_dice16_ecs215_psi2
```

********************************************************************************

**DICE-2016, psi=2., ECS=4.55:**

To run the model:

```
  - constants: dice_generic_2016_ecs455_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_dice16_ecs455_psi2
```

To replicate the neural network training:
Use the solution to the DICE-2016 model with psi=2. as a pretrained neural network

********************************************************************************

**CDICE, psi=2.:**

To run the model:

```
  - constants: dice_generic_mmm_mmm_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_mmm
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_mmm_mmm_psi2
```

To replicate the neural network training:
Use the solution to the CDICE model as a pretrained neural network

********************************************************************************

**CDICE-GISS-E2-R, psi=2.:**

To run the model:

```
  - constants: dice_generic_mmm_giss_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_giss
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_mmm_giss_psi2
```

To replicate the neural network training:
Use the solution to the CDICE-GISS-E2-R model as a pretrained neural network

********************************************************************************

**CDICE-HadGEM2-ES, psi = 2.:**

To run the model:

```
  - constants: dice_generic_mmm_hadgem_psi2
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_hadgem
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES2/Opt_mmm_hadgem_psi2
```

To replicate the neural network training:
Use the solution to the CDICE-HadGEM2-ES model as a pretrained neural network

********************************************************************************

**DICE-2016, psi=0.5:**

To run the model:

```
  - constants: dice_generic_2016_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_dice16_psi05
```

To replicate the neural network training:
Use the solution to the DICE-2016 model as a pretrained neural network; train the
model with psi=0.6 and rho=0.01 for the episodes 101-150; train the
model with psi=0.57 and rho=0.008 for the episodes 151-200; train the
model with psi=0.55 and rho=0.006 for the episodes 201-275; train the
model with psi=0.52 and rho=0.005 for the episodes 276-375; train the
model with psi=0.5 and rho=0.004 from the episode 376 on;  

********************************************************************************

**DICE-2016, psi=0.5, ECS=4.55:**

To run the model:

```
  - constants: dice_generic_2016_ecs455_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_dice16_ecs455_psi05
```
To replicate the neural network training:
Use the solution to the DICE-2016 model with psi=0.5 as a pretrained neural network;

********************************************************************************

**DICE-2016, psi=0.5., ECS=2.15:**

To run the model:

```
  - constants: dice_generic_2016_ecs215_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_2016
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_dice16_ecs215_psi05
```

To replicate the neural network training:
Use the solution to the DICE-2016 model with psi=0.5 as a pretrained neural network;

********************************************************************************

**CDICE, psi=0.5:**

To run the model:

```
  - constants: dice_generic_mmm_mmm_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_mmm
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_mmm_mmm_psi05
```

To replicate the neural network training:
Use the solution to the CDICE model as a pretrained neural network; train the
model with psi=0.6 and rho=0.01 for the episodes 151-200; train the
model with psi=0.55 and rho=0.007 for the episodes 200-300; train the
model with psi=0.5 and rho=0.004 from the episode 301 on;  

********************************************************************************

**CDICE-GISS-E2-R, psi=0.5:**

To run the model:

```
  - constants: dice_generic_mmm_giss_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts
  - variables: dice_generic_mmm_giss
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_mmm_giss_psi05
```

To replicate the neural network training:
Use the solution to the CDICE model with psi=0.5 as a pretrained neural network;

#Note
The solution for the model CDICE-GISS-E2-R with psi=0.5 is large, thus it is stored in the archive. Before the prostrpocessing go to the folder ```<PATH_TO_THE_FOLDER>/Climate_in_Climate_Economics/DEQN_for_IAMS/dice_generic/optimal_results/IES05/Opt_mmm_giss_psi05``` and unpack the archive:

```
$ tar -xf NN_data.tar.xz
```
To run the postprocessing make sure that the content of the folder ```NN_data``` is stored in the respective folder of the model, otherwise, one needs to modify a postprocessing path ```  <MODEL_FOLDER>```.

********************************************************************************

**CDICE-HadGEM2-ES, psi = 0.5:**

To run the model:

```
  - constants: dice_generic_mmm_hadgem_psi05
  - net: dice_generic
  - optimizer: dice_generic
  - run:  dice_generic_1yts   
  - variables: dice_generic_mmm_hadgem
```
To postprocess:

```
  <MODEL_FOLDER> = dice_generic/optimal_results/IES05/Opt_mmm_hadgem_psi05
```
To replicate the neural network training:
Use the solution to the CDICE-HadGEM2-ES model as a pretrained neural network; train the
model with psi=0.6 and rho=0.01 for the episodes 151-210; train the
model with psi=0.55 and rho=0.007 for the episodes 211-250; train the
model with psi=0.5 and rho=0.004 from the episode 251 on;

#Note
The solution for the model CDICE-HadGEM2-ES with psi=0.5 is large, thus it is stored in the archive. Before the prostrpocessing go to the folder ```<PATH_TO_THE_FOLDER>/Climate_in_Climate_Economics/DEQN_for_IAMS/dice_generic/optimal_results/IES05/Opt_mmm_hadgem_psi05``` and unpack the archive:

```
$ tar -xf NN_data.tar.xz
```
To run the postprocessing make sure that the content of the folder ```NN_data``` is stored in the respective folder of the model, otherwise, one needs to modify a postprocessing path ```  <MODEL_FOLDER>```.
