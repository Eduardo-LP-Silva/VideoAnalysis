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
Through the analysis of the evaluation metrics, we can see that the model achieved an acceptable performance. However, there are certain groups where this is not the case, such as in the "People & Family & Pets" topic, which has the lowest F-1 score of all the topics due to its poor precision. Perhaps it would be useful to separate this topic into multiple, more precise labels, as even the name implies a somewhat high variability in content.

At first glance, the average intra-group cosine similarity doesn't seem to be correlated with performance, as, for instance, the topic "TechConsumerElectronics" has a lower F-1 score than "Sports" and "Fashion" despite these topics having a higher and lower average cosine similarity, respectively.

We can also see that the evaluation metrics of the video's topic groups have a higher range than the video's TV show groups, the latter generally having a higher performance, indicating that the model is more consistent in identifying the correct topic within the same show than across multiple shows of the same topic.

Finally, if the PCA values for each entry (obtained through the API by reducing the feature vector of each one) are plotted in a 2D scatter chart, we can also see that most records are packed closely together without any particular boundaries between topics, while the tv shows have more separability between them.
