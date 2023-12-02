## 流程
### 1. 定义BlockSolverType
1. `g2o::BlockSolver<Traits>` 是一个模板类型，需要传入一个 `g2o::BlockSolverTraits<int, int>` 类型。
2. `g2o::BlockSolverTraits<int, int>` 也是个模板类型，要用 `template <int _PoseDim, int _LandmarkDim>` 明确类型。
3. `_PoseDim` 是优化变量纬度； `_LandmarkDim` 是误差纬度?
```cpp
//示例
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
```
4. `g2o::BlockSolverTraits<int, int>` 里面定义了各种类型，包括 `PoseMatrixType`, `LinearSolverType` 等等。这些类型都会传给 `BlockSolverType` 。
``` cpp
// g2o::BlockSolverTraits<int _PoseDim, int _LandmarkDim> 中的两个模板变量，以及type定义
static const int PoseDim = _PoseDim; //通过模板参数确定
static const int LandmarkDim = _LandmarkDim; //通过模板参数确定
typedef Eigen::Matrix<double, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType;
typedef LinearSolver<PoseMatrixType> LinearSolverType

// BlockSolver<typename Traits>中的 两个模板变量，以及type定义，他们都是来源于g2o::BlockSolverTraits<int _PoseDim, int _LandmarkDim>
static const int PoseDim = Traits::PoseDim;
static const int LandmarkDim = Traits::LandmarkDim;
typedef typename Traits::PoseMatrixType PoseMatrixType;
typedef typename Traits::LinearSolverType LinearSolverType;
```

### 2. 定义LinearSolverType
1. `LinearSolverType` 是一个`LinearSolver<MatrixType>`类型。这里的`MatrixType`需要是上文定义的`PoseMatrixType`类型，也就是`typedef Eigen::Matrix<double, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType`。
2. `LinearSolverType`已经在`BlockSolverType`里面已经有定义了，如上文。
```cpp
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
BlockSolverType::LinearSolverType temp;
```
3. 可以自定义 `LinearSolverType`，如：`LinearSolverDense<MatrixType>`, `LinearSolverCSparse<MatrixType>`
```cpp
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
```

### 3. 实例化BlockSolverType
1. 前面定义好了各种type，`BlockSolverType`的实例化，要传入一个`LinearSolverType`的unique_ptr。
```cpp
//BlockSolver 的构造函数
BlockSolver(std::unique_ptr<LinearSolverType> linearSolver);
// 实例化BlockSolver
auto blocksv = g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
```

### 4. 实例化 OptimizationAlgorithm{GaussNewton、Levenberg、Dogleg}类
1. 以`OptimizationAlgorithmGaussNewton`为例，实例化需要传入一个`std::unique_ptr<Solver> solver`类型的类。而`BlockSolver`继承了 `BlockSolverBase`, `BlockSolverBase` 继承了 `Solver` ， 所以`BlockSolver`是一个`Solver`，可用我们实例化的`BlockSolverType`进行初始化，如下：
```cpp
//OptimizationAlgorithmGaussNewton 的构造函数
OptimizationAlgorithmGaussNewton(std::unique_ptr<Solver> solver);
//定义总solver
auto blocksv = g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
auto solver = new g2o::OptimizationAlgorithmGaussNewton(blocksv);
```
### 5. 实例化SparseOptimizer类
```cpp
g2o::SparseOptimizer optimizer;     // 图模型
optimizer.setAlgorithm(solver);   // 设置求解器
optimizer.setVerbose(true);       // 打开调试输出
```

## 顶点
### 1. g2o::BaseVertex&lt;int, typename T&gt;
1. `BaseVertex<int D, typename T>` 是一个模板类，继承自`OptimizableGraph::Vertex`。模板参数含义为：`<顶点的纬度, 顶点的类型>`。
2. `BaseVertex`定义了以下类型和变量。
```cpp
using EstimateType = T;
using BackupStackType = std::stack<EstimateType, std::vector<EstimateType> >;
static const int Dimension = D;  ///< dimension of the estimate (minimal) in the manifold space
```
3. 有以下成员变量,其中比较重要的是`_estimate`，这个是顶点的值。
```cpp
HessianBlockType _hessian;
Eigen::Matrix<double, D, 1, Eigen::ColMajor> _b;
EstimateType _estimate;
BackupStackType _backup;
```
4. `BaseVertex<int D, typename T>`有`void setEstimate(const EstimateType& et)`函数，用来对`_estimate`设置初值。还有`const EstimateType& estimate()`函数，返回顶点当前`_estimate`的值。
```cpp
//! return the current estimate of the vertex
const EstimateType& estimate() const { return _estimate; }
void setEstimate(const EstimateType& et) {
  _estimate = et;
  updateCache();
}
```

### 2. OptimizableGraph::Vertex
1. `OptimizableGraph::Vertex`继承自`HyperGraph::Vertex`和`HyperGraph::DataContainer`。
2. `OptimizableGraph::Vertex`有纯虚函数`setToOriginImpl()`，需要在自己的顶点定义去实现。用来对`BaseVertex<int D, typename T>`中的`_estimate`设初值。
```cpp
//! sets the node to the origin (used in the multilevel stuff)
virtual void setToOriginImpl() = 0;
```
3. `OptimizableGraph::Vertex`有纯虚函数`oplusImpl(const double* v)`，需要在自己的顶点定义去实现。用来对`BaseVertex<int D, typename T>`中的`_estimate`进行更新。其中v为更新量，纬度和顶点的纬度一致？
```cpp
/**
  * update the position of the node from the parameters in v.
  * Implement in your class!
  */
virtual void oplusImpl(const double* v) = 0;
```
4. `OptimizableGraph::Vertex`有纯虚函数`read(std::istream& is)`和`write(std::ostream& os)`，需要在自己的顶点定义去实现,可以写一个空函数。
```cpp
//! read the vertex from a stream, i.e., the internal state of the vertex
virtual bool read(std::istream& is) = 0;
//! write the vertex to a stream
virtual bool write(std::ostream& os) const = 0;
```
5. `OptimizableGraph::Vertex`有函数虚函数`virtual void setId(int id)`,它已经帮我们实现了。用于设置`_id`。`_id`是`HyperGraph::Vertex`的一个成员变量。
```cpp
//! sets the id of the node in the graph be sure that the graph keeps
//! consistent after changing the id
virtual void setId(int id) { _id = id; }
```
5. `OptimizableGraph::Vertex`定义了以下成员变量
```cpp
OptimizableGraph* _graph;
Data* _userData;
int _hessianIndex;
bool _fixed;
bool _marginalized;
int _dimension;
int _colInHessian;
OpenMPMutex _quadraticFormMutex;

CacheContainer* _cacheContainer;
```
### 3. HyperGraph::Vertex
1. `HyperGraph::Vertex`继承自`HyperGraphElement`
2. 有以下成员变量
```cpp
int _id;
EdgeSet _edges;
//EdgeSet 的 定义
typedef std::set<Edge*> EdgeSet;
```

## 边

### 1. BaseUnaryEdge&lt;int D, typename E, typename VertexXi&gt;
1. `g2o::BaseUnaryEdge<int D, typename E, typename VertexXi>`是一元边的模板类，模板参数分别是`<误差纬度， 误差类型， 连接顶点类型>`，继承自`BaseFixedSizedEdge<int D, typename E, typename... VertexTypes>`。`BaseUnaryEdge`的模板参数`VertexXi`会传给`BaseFixedSizedEdge`的`VertexTypes`。
2. 需要注意的是引用`_jacobianOplusXi`，需要我们自己计算填写。
```cpp
using VertexXiType = VertexXi;
BaseUnaryEdge() : BaseFixedSizedEdge<D, E, VertexXi>(){};

protected:
typename BaseFixedSizedEdge<D, E, VertexXi>::template JacobianType<  //_jacobianOplusXi是_jacobianOplus[0]的一个引用
    D, VertexXi::Dimension>& _jacobianOplusXi =
    std::get<0>(this->_jacobianOplus);

// typename：在模板代码中，当依赖于模板参数的类型被引用时，需要使用 typename 关键字来指示该名称是一个类型。
// BaseFixedSizedEdge<D, E, VertexXi>：这是一个类模板的实例化，其中的类型参数为 D、E 和 VertexXi。这个实例化的类型是 BaseFixedSizedEdge。
// template JacobianType<D, VertexXi::Dimension>：这是 BaseFixedSizedEdge 类中的一个嵌套类型 JacobianType 的实例化，其中的类型参数为 D 和 VertexXi::Dimension。
// VertexXi::Dimension：VertexXi 是一个类型，它具有一个叫做 Dimension 的静态成员变量。
// std::get<0>(this->_jacobianOplus)：std::get 是一个函数模板，用于从元组（tuple）中获取指定索引的元素。this->_jacobianOplus 是一个元组，这里使用 std::get<0> 来获取元组中的第一个元素。
// 综上所述，这段代码的作用是将 this->_jacobianOplus 中的第一个元素赋值给 _jacobianOplusXi，并且 _jacobianOplusXi 的类型是通过类模板实例化和嵌套类型实例化得到的。
```

### 2. BaseFixedSizedEdge&lt;int D, typename E, typename... VertexTypes&gt;
1. `BaseFixedSizedEdg<int D, typename E, typename... VertexTypes>`继承自`BaseEdge<int D, typename E>`。
2. `BaseFixedSizedEdg<int D, typename E, typename... VertexTypes>` 定义了以下几种type和变量：
```cpp
static const int Dimension = BaseEdge<D, E>::Dimension;
typedef typename BaseEdge<D, E>::Measurement Measurement;
typedef typename BaseEdge<D, E>::ErrorVector ErrorVector;
typedef typename BaseEdge<D, E>::InformationType InformationType;

template <int EdgeDimension, int VertexDimension>
using JacobianType = typename Eigen::Matrix<
    double, EdgeDimension, VertexDimension,
    EdgeDimension == 1 ? Eigen::RowMajor : Eigen::ColMajor>::AlignedMapType;


HessianRowMajorStorage _hessianRowMajor;
HessianTuple _hessianTuple;
HessianTupleTransposed _hessianTupleTransposed; //这几个Hessian开头的类型都和VertexTypes有关
std::tuple<JacobianType<D, VertexTypes::Dimension>...> _jacobianOplus; //这个是雅克比矩阵
// <JacobianType<D, VertexTypes::Dimension>...>：使用了展开包的语法 ...，表示元组中的元素类型是多个 JacobianType<D, VertexTypes::Dimension> 的实例。
```
3. 定义了纯虚函数`void linearizeOplus()`，需要在自己定义的边里实现，用来计算雅克比矩阵`BaseUnaryEdge`中的`_jacobianOplusXi`，实际上是设置自己的`_jacobianOplus`，因为`_jacobianOplusXi`是`_jacobianOplus`的第一个元素的引用。不知道这里的`linearizeOplus`函数和 `OptimizableGraph::Edge`中的`linearizeOplus`的区别是什么？
```cpp
/**
  * Linearizes the oplus operator in the vertex, and stores
  * the result in temporary variables _jacobianOplus
  */
virtual void linearizeOplus();
```

### 3. BaseEdge&lt;int D, typename E&gt;
1. `BaseEdge<int D, typename E>`继承自`OptimizableGraph::Edge`。构造时会将`OptimizableGraph::Edge`的`_dimension`设为D。
```cpp
BaseEdge() : OptimizableGraph::Edge() { _dimension = D; }
```
2. `BaseEdge<int D, typename E>`定义了以下几种类型
```cpp
static constexpr int Dimension = internal::BaseEdgeTraits<D>::Dimension;
typedef E Measurement;
typedef typename internal::BaseEdgeTraits<D>::ErrorVector ErrorVector;
typedef typename internal::BaseEdgeTraits<D>::InformationType InformationType;
```
3. `BaseEdge<int D, typename E>`用了`BaseEdgeTraits<D>`， 和顶点一样， `BaseEdgeTraits` 定义了三种类型，并传给了`BaseEdge` 。
```cpp
// BaseEdgeTraits 的定义
template <int D>
struct BaseEdgeTraits {
  static constexpr int Dimension = D;
  typedef Eigen::Matrix<double, D, 1, Eigen::ColMajor> ErrorVector;
  typedef Eigen::Matrix<double, D, D, Eigen::ColMajor> InformationType;
};
```
4. `BaseEdge<int D, typename E>` 有以下几种成员变量
```cpp
Measurement _measurement;      ///< the measurement of the edge
InformationType _information;  ///< information matrix of the edge.
                                ///< Information = inv(covariance)
ErrorVector _error;  ///< error vector, stores the result after computeError()
                    ///< is called
```
5. 定义并实现了虚函数 `setMeasurement(const Measurement& m)`，用来设置`_measurement`变量。`_measurement`就是观测值。
```cpp
virtual void setMeasurement(const Measurement& m) { _measurement = m; }
```
6. 定义并实现了函数 `setInformation(const InformationType& information)`,用来设置`_information`信息矩阵变量。
```cpp
void setInformation(const InformationType& information) {
  _information = information;
}
```

### 4. OptimizableGraph::Edge
1. `OptimizableGraph::Edge`继承自`HyperGraph::Edge`和`HyperGraph::DataContainer`。
2. `OptimizableGraph::Edge` 中有以下成员变量
```cpp
int _dimension;
int _level;
RobustKernel* _robustKernel;
long long _internalId;
std::vector<int> _cacheIds;
std::vector<std::string> _parameterTypes;
std::vector<Parameter**> _parameters;
std::vector<int> _parameterIds;
```
3. 定义了纯虚函数`void computeError()`，需要在自己定义的边里实现。用以计算`BaseEdge<int D, typename E>`里的`_error`。具体做法是a.取出这条边的顶点的估计值。b.根据模型计算估计观测量。c.用估计观测量值和`_measurement`做差，将差填入`_error`。
```cpp
// computes the error of the edge and stores it in an internal structure
virtual void computeError() = 0;
```
4. 定义了纯虚函数`void linearizeOplus(JacobianWorkspace& jacobianWorkspace)`，需要在自己定义的边里实现，用来计算雅克比矩阵。
```cpp
/**
  * Linearizes the constraint in the edge in the manifold space, and store
  * the result in the given workspace
  */
virtual void linearizeOplus(JacobianWorkspace& jacobianWorkspace) = 0;
```

### 5. HyperGraph::Edge
1. `HyperGraph::Edge` 定义了以下变量
```cpp
// HyperGraph::Edge
VertexContainer _vertices;
int _id;  ///< unique id
// VertexContainer的定义
typedef std::vector<Vertex*> VertexContainer;
```
2. 提供函数`setId(int id)`，用于设置自己的id。
```cpp
void HyperGraph::Edge::setId(int id) { _id = id; }
```
3.提供函数`void setVertex(size_t i, Vertex* v)`,往`_vertices`index为`i`的位置装顶点，也就是这条边连接的顶点。
```cpp
void setVertex(size_t i, Vertex* v) {
  assert(i < _vertices.size() && "index out of bounds");
  _vertices[i] = v;
}
```


## curvefitting 代码详解
### 1. 流程
1. 用 BlockSolver 定义一个总的 solver
2. 定义 SparseOptimizer optimizer 总优化器，设置 optimizer.setAlgorithm(solver)、.setVerbose
3. 生成顶点 optimizer.addVertex
4. 生成边 optimizer.addEdge
5. optimizer.initializeOptimization();
6. optimizer.optimize(10);
7. 取优化数值 estimate = v->estimate();
### 2. 顶点
1. 自定义一个类 CurveFittingVertex ，继承 BaseVertex; BaseVertex 要给定顶点纬度D和类型T。
2. CurveFittingVertex 需要重写 setToOriginImpl()， oplusImpl(const double *update)， read和write方法。
3. setToOriginImpl对_estimate设置初值，它是一个T类型；oplusImpl中的update是增量，纬度是D，将update加到_estimate。
4. v->setEstimate(const EstimateType& et)，对_estimate设初值；v->setId(int id)，设置顶点id。
### 3. 边
1. 自定义CurveFittingEdge，继承 BaseUnaryEdge； BaseUnaryEdge 要给定 误差纬度D，误差类型T和顶点类型。
2. 设置边的id；设置连接的顶点（将顶点放在 _vertices 的哪个位置）；设置Measurement, 即_measurement的值；设置_information。
3. 重写 computeError()、linearizeOplus、read和write方法。
4. computeError，取_vertices中的顶点（位置由前决定）；获取顶点估计值est = v->estimate();计算误差 _measurement和估计值y的误差 _error。_error是<double, D,1>类型的。
5. linearizeOplus，计算雅克比。取_vertices中的顶点（位置由前决定）；获取顶点估计值est = v->estimate()，计算估计值y。填入雅克比 _jacobianOplusXi。雅克比的类型是 `  using JacobianType = typename Eigen::Matrix<double, EdgeDimension, VertexDimension,
EdgeDimension == 1 ? Eigen::RowMajor : Eigen::ColMajor>::AlignedMapType;`



