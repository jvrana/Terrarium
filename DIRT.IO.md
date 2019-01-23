DIRT.IO Design format

A design is simply a **Model** (probability graph of all operation types) and a **Sample Composition** which defines the goals of the design.

```json

```

#### Model

#### Sample Composition

A sample composition object is a nested query object comprised of the 

```
{
    "sample": [{Query}],
    "children": [{Sample Composition}],
    "filters": {[Query]}
}
```

##### Query

The query is the fundamental unit to access and represent samples, object types and inventory in **DIRT.IO**. 

The following queries a database for any samples of the "Yeast Type" owned by "Justin Vrana"

```json
{
    "__typename": "Sample",
    "sample_type": {
        "__typename": "SampleType",
        "name": "Yeast Strain"
    },
    "user": {
        "__typename": "User",
        "name": "Justin Vrana"
    }
}
```

The following query finds any inventory of a Yeast Glycerol Stock of a particular set of samples

```json
{
    "__typename": "Item"
    "object_type": {
        "__typename": "ObjectType",
        "name": "Yeast Glycerol Stock",
    },
    "sample": {
        "__typename": "Sample",
        "name": ["3NodeSwitch_G1", "3NodeSwitch_G2", "3NodeSwitch_G3"]
    }
}
```



### Exception Handler

Exception handlers are how Terrarium handles certain events during the normal (or abnormal) execution of a plan. Handlers consist of an **event** and an **action**. Event emitters run automatically to emit certain events.cke

```
{
    "event": null,
    "action": [???]
}
```

### Event

```
{
    
}
```

### Action

```
{
    
}
```

