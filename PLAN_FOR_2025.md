# Tasks

- validate inputs 
    - POST /aimodels
    - POST /engines
    - POST /generators
    - POST /jobs

- endpoints for deletion
  - DELETE /aimodels/:id
    - can't delete if is in engine
  - DELETE /engines/:id
    - cen't delete if is in generator
  - DELETE /generators/:id
    - deletes all jobs also
    - can't delete if it is not closed
  - DELETE /jobs/:id
    - delete its images and poses
    - can't delete while processing

- endpoints for images
  - get /images 
  - get /images/:id  (will print info)
  - get /images/:id/show  (will show the photo if it is ready or else will return an error)
