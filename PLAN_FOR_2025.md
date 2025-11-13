# Tasks

- before start generator check that it is something else other than 'closed' or else return error 
- on closing generator wait until job finish


- Implement generator's MVP REST API:   
    - DELETE /generators/{id}
    - POST /generators/{id}/jobs adds a jobs to DB
      - ProcessManager will have a thread that will pull jobs from the DB to process when the generator is ready to accept jobs
    - GET /generators/{id}/jobs

- create tests to implement and check validations for POST /aimodels, /engines, /generators and /jobs
- create tests for all generator's actions



## Done
### Nov 11 2025
- Implement generator's MVP REST API:
    - POST /generators/ that takes engine's id as input to create the generator and responds with generator's ID
    - GET /generators/ and GET /generators/{id} that returns generator's status
    - PATCH /generators/{id}/start starts the generator in a multiprocess
    - PATCH /generators/{id}/close to close the generator

- Implement engine's MVP REST API:
    - POST /engines/ ✅☑
    - GET /engines/{id}
    - GET /engines/
