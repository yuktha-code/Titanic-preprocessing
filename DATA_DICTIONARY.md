# Data Dictionary

## Original columns

| Column | Meaning |
|---|---|
| PassengerId | Unique passenger identifier |
| Survived | Target: 0 = did not survive, 1 = survived |
| Pclass | Ticket class: 1 = first, 2 = second, 3 = third |
| Name | Passenger name |
| Sex | Passenger sex recorded in the dataset |
| Age | Age in years |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port: C = Cherbourg, Q = Queenstown, S = Southampton |

## Engineered columns

| Column | Construction and purpose |
|---|---|
| Title | Title extracted from Name and grouped into Mr, Miss, Mrs, Master, or Rare |
| FamilySize | `SibSp + Parch + 1` |
| IsAlone | 1 when FamilySize is 1, otherwise 0 |
| TicketGroupSize | Number of passengers sharing the same Ticket value |
| FarePerPerson | Fare divided by TicketGroupSize |
| CabinKnown | 1 when Cabin is present, otherwise 0 |
| Deck | First character of Cabin; missing cabins become Unknown |

The fully preprocessed table one-hot encodes Pclass, Sex, Embarked, Title, and
Deck. It standardizes Age, SibSp, Parch, Fare, FamilySize, TicketGroupSize, and
FarePerPerson. Binary fields remain 0/1.

