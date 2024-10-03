# Video Analysis API
This is a simple FastAPI app intended to parse and analyse a small dataset of video information (in CSV format) while using MongoDB as storage.

It exposes two endpoints, one for obtaining the videos' information with a dimensionally reduced feature vector, and another to obtain statistics regarding several groupings. Both endpoints can be supplied with optional filters. For more information on the values expected by them, the schemas part of the Swagger UI docs can be consulted.

## How to run
To run the project you will need [Docker](https://docs.docker.com/engine/install/). To start the app, you can simply use the command:
```shell
docker compose up
```

You can then access the Swagger UI at localhost:8080/docs.

## Dataset Analysis
A general analysis of the dataset reveals that:
* There are 3 topics with a significantly higher number of associated videos (People & Family & Pets, FoodBev, Health Care).
* There are 3 TV shows with a significantly higher number of entries (s10f, g0a, t0a).
* The feature vectors do not have unit norm.
* There can be multiple topics per show.
* There are 50 entries which have a predicted label of "Undefined". This could mean several things, such as the existence of another category of the same name but not present in this particular set's actual labels.

### Further insights
Through the analysis of the evaluation metrics, we can see that the model achieved an acceptable performance. However, there are certain groups where this is not the case, such as in the "People & Family & Pets" and "Events and attractions" topics, which have the lowest F-1 scores of all the topics, possibly due to their broad range of content, as their name implies. Perhaps it would be useful to split these topics into additional, more focused ones.

As for the average intra-group cosine similarity, we can see a (weak) inverse trend between it and the F-1 score for each topic, i.e., as it increases, the performance seems to drop, which is not what one would expect. Perhaps as the intra-topic similarity increases, so too does the inter-topic one, which hinders the model's performance. Note that not every topic group follows this trend, however.

The opposite can be seen between the TV shows' intra-group cosine similarities and their F-1 values (the latter increases with the former), which is somewhat expected as the lower content cohesiveness within a group would make it harder for a model to correctly predict the correct topic.

Finally, if the PCA values for each entry (obtained through the API by reducing the feature vector of each one) are plotted in a 2D scatter chart, we can also see that most records are packed closely together without any particular boundaries between topics, while the tv shows have more separability between them.
