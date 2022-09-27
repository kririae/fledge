There're exactly four types of classes that we might want to handle. 
Note that ALL classes must implement constructor.

1. The most trivial type of classes, implemented only on CPU, i.e., `Render` (1)
2. ALL functions including ctors and dtors are implemented on both sides, i.e., `Vector3f, Ray` (2)
3. All functions except ctors and dtors are implemented on both sides, i.e., `Light, MaterialDispatcher` (3)
4. All functions except ctors and dtors are implemented only on GPU, `CUDATexture` (4). 
In this type of class, we might expect that there's a corresponding CPU version.

We have two types of aggregation.

1. type $A$ contains an instance of type $B$.
2. type $A$ contains a pointer to type $B$.

We list a series of difficulties here.
