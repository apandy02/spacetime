# spacetime 

![Minkowski diagram](minkowski.png)


learning algorithms for video tasks. 

## Models

- [Genie](src/spacetime/models/genie): Generative Interactive Environments, Bruce et. al. 2024
  - [Model](src/spacetime/models/genie/model.py): Model implementation -- wrapper that puts all the components together for joint training (latent action model and dynamics model as well as a prettained tokenizer)
  - [Dynamics Model](src/spacetime/models/genie/dynamics.py): Dynamics model implementation
  - [Latent Action Model](src/spacetime/models/genie/latent_actions.py): Latent action model implementation 
