# Tasks


- Implement generator's MVP REST API:
    - [20%] PATCH /generators/{id}/start starts the generator in a multiprocess
    - [20%] PATCH /generators/{id}/stop to stop the generator and delete the process, it will wait until job finish
    - PATCH /generators/{id}/stop/force to stop the generator and delete the process, it will NOT wait for the job finish
    - DELETE /generators/{id}
    - DELETE /generators/{id}/force to stop jobs and delete the generator

  - implement job's MVP rest api
    - POST /jobs adds a jobs to start processing
      - if generator is full, keep it in a queue
      - if generator doing nothing, start the job and when finish save image
    - GET /jobs

- create tests to implement and check validations for POST /aimodels, /engines, /generators and /jobs
- create tests for all generator's actions



## Done
### Nov 11 2025
- Implement generator's MVP REST API:
    - POST /generators/ that takes engine's id as input to create the generator and responds with generator's ID
    - GET /generators/ and GET /generators/{id} that returns generator's status

- Implement engine's MVP REST API:
    - POST /engines/ ✅☑
    - GET /engines/{id}
    - GET /engines/
