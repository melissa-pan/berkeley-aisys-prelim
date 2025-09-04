# Why Systolic Architectures

## 📋 Basic Information
- **Authors**: H.T. Kung
- **Year**: 1982
- **Venue**: Computer (IEEE)
- **Link**: [PDF](http://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf)



# 1. 📖 Paper Understanding

## The Problem

### What problem does this paper solve?

ASIC are being constrcuted, but mostly ad hoc - IO computation imbalance is a problem in many of these systems, which only come to be discovered after the ASIC is being built out.

- How to systematically design cost-effective, high-throughput special purpose systems for regular (repetitive) compute-bound tasks without being throttled by memory I/O bandwidth
- Need for high-performance computing systems that can exploit parallelism effectively while balancing computation with I/O operations


### Prior art and why they didn't work well:
Many special-purpose systems were ad hoc, lacking a general methodology; repeated engineering mistakes (e.g., discovering I/O limits after building the device).


## The Key Idea

A general methodology for mapping high level computations into hardware structures.

Pro:
- easy to implement because of regularity
- easy to reconfigure due to its modularity

Systolic architectures are networks of processors that rhythmically compute and pass data through the system, like blood flowing through arteries (hence "systolic"):
- Regular, predictable data flow patterns with synchronized processing
- Local communication between neighboring processors only
- Pipelined execution where multiple computations proceed simultaneously
- Match algorithm structure directly to hardware architecture for optimal efficiency
- PRO: higher computation throughput without increasing memory bandwidth

Diminishing growth rate for component speed up, need for concurrency where coordination and communication becoems significant.

## The Challenge
### What are the main challenges in solving this problem?
- **Synchronization complexity**: Ensuring data arrives at the right time at each processing element
- **Load balancing**: Keeping all processors busy with useful work throughout execution
- **I/O bandwidth**: Balancing computation rate with available I/O bandwidth to avoid bottlenecks
- **Algorithm mapping**: Restructuring existing algorithms to fit systolic execution patterns
- **Cost-effectiveness**: Creating systems that are both high-performance and economically viable

## The Method
### Brief overview (detailed analysis in Section 2):
- Network of identical processing elements (PEs) arranged in regular, modular patterns
- Data flows through the array in a synchronized, rhythmic fashion with global timing
- Each processor performs simple operations on streaming data with local memory
- Results propagate through the network following predictable paths to produce output
- Multiple computations per I/O access to optimize bandwidth utilization

## Pros & Cons
### Strengths:
- **High throughput**: Pipelined execution enables multiple computations simultaneously
- **Cost-effectiveness**: Simple, regular design reduces both design and manufacturing costs  
- **Scalability**: Modular structure allows easy expansion to meet performance requirements
- **VLSI-friendly**: Regular, repetitive structure ideal for integrated circuit implementation
- **Predictable performance**: Regular data flow patterns enable accurate performance analysis
- **Energy efficiency**: Local communication significantly reduces power consumption

### Weaknesses/Limitations:
- **Limited applicability**: Works optimally only for regular, repetitive computational patterns
- **Programming complexity**: Requires significant algorithm restructuring to fit systolic model
- **Input/output constraints**: Edge processors may become bottlenecks for data flow
- **Synchronization overhead**: Requires precise global timing and clock distribution
- **Specialization**: Design is often specific to particular classes of problems

## Impact & Contributions
### Key contributions to the field:
- **Architectural paradigm**: Introduced systematic methodology for designing parallel special-purpose systems
- **Algorithm-architecture co-design**: Demonstrated importance of matching hardware structure to computation patterns
- **VLSI design principles**: Showed how to effectively utilize large-scale integration technology
- **Performance-cost optimization**: Provided framework for balancing performance with implementation cost
- **Foundation for modern accelerators**: Laid conceptual groundwork for specialized computing architectures

The vision: "The challenge is to develop architectures that can deliver the computational power of many processors while maintaining cost-effectiveness and programming simplicity through regular, modular designs."

### How did this paper change the field after its release?
- **AI hardware revolution**: Modern neural network accelerators (Google TPU, etc.) directly implement systolic principles
- **GPU architecture influence**: Graphics processors adopted similar pipelined, parallel execution models
- **Signal processing systems**: Became standard approach for DSP and real-time signal processing hardware
- **Scientific computing**: Influenced design of high-performance computing systems and supercomputers  
- **FPGA implementations**: Systolic designs became popular pattern in reconfigurable computing


## Background & History
- **Useful background knowledge:**
  - VLSI design principles and silicon area constraints
  - Parallel computing models and pipeline processing
  - Matrix algorithms and linear algebra computations
  - Computer architecture and memory hierarchy design

### **Pre-history and context:**
- Late 1970s-early 1980s: VLSI technology revolution enabling massive integration
- Growing computational demands in scientific computing and signal processing
- Need for specialized architectures to effectively utilize silicon area
- Recognition that general-purpose processors were insufficient for compute-intensive tasks



# 2. 🔬 Key Technical Details

## Method
- **Processing Elements (PEs)**: Simple processors with local memory, arithmetic units, and communication interfaces
- **Interconnection Pattern**: Regular grid, linear array, or tree structure with nearest-neighbor communication
- **Data Movement**: Synchronized, rhythmic data flow with predictable timing relationships
- **Control Structure**: Global clock synchronization with distributed local control logic
- **Memory Organization**: Hierarchical system with local PE memory and external I/O interfaces

## half-systolic
con: save pin than full systollic

### B1: broadcast inputs, move results, weight stays

At each clock cycle:
- A new input Xj is broadcast to all cells simultaneously.
- A new partial sum (initialized to zero) enters the leftmost cell.

Each cell does:
- `y <- y + (wi . xj)`
- then forward y to next cell

pro:
- regular pipeline structure, same PE everywhere
- use local communication for results
- high throughput: 1 result per cycle after warmup

con:
- requires global braodcast of xj -> long buses/wires are costly to route and scale
- yi move along the array, so wide datapath are needed. -> y is usually more bits than w

### B2: braodcast inputs, move weights, results stays
At each clock cycle:
- A new input Xj is broadcast to all cells simultaneously.
- each output Yj accumulated and stay in the cell
- weights move systocally across the array

Pro:
- each cell needs a single multiplier accumulator -> indicate when to output the y and resets the value
- output don't move: save bandwidth
- increase precision: keep an extra guard bit

Con:
- Still requires a global input broadcast
- Need extra bus for output collection

Design B2 is closer to how many modern DSPs and accelerators work.

### Design F: Fan-in result (with a seperate adder), pipeline inputs, weight stays
Each cycle:
- element xi move one step to the right
- each cell multiplies its weight by the current x
- all product are fanned into an adder tree to produce yi per-cycle after the pipeline fills

pro:
- natural match to dot product
- high throughput
- if k is large, can still pipeline the adder tree -> still fast

con:
- requires a global fan-in network -> less modular
- as K grow, fan-in wiring and tree depth grow -> potential scaling issue

## Pure Systolic

### R1 — Results stay, inputs and weights move in opposite directions

- Each cell holds an accumulator for one output yi
- weights: streams from right to left
- inputs (x): stream from left to right
- results: stays
- Weights and inputs are each seperated by 1 cycles to ensure all xi meets all wi

Pro:
- result stays in place, same benefit as B2

Con: 
- only 50% utilization, but can interleave two inner product with an additional accumulator in each cell

### R2 - Results stay, inputs and weights move in the same direction but at different speeds
- Y: stay in cell
- x an w move in the same direction
- x 2x faster by storing w for an extra round in each cell

Pro: All cells busy every cycle → better hardware utilization than R1.

Con: require an extra regier per cell to store the weights


### W1 — Weights stay, inputs and results move in opposite directions
- weight stay fixed in each cell
- input comes in left to right
- result comes out right to left

pro: 
- results stream out naturally in order
- no need for a seperate output bus
- response time is constant
- support 2 level pipelining

con:
- like R1, only 50% compute utilization

### W2 — Weights stay, inputs and results move in the same direction (different speeds)
- weight stays
- input left to right
- output left to right (but more slow, or vice versa)
- support 2 level pipelining

pro:
- All cells are busy each cycle → high throughput.
- Can produce one new result per cycle once the pipeline fills.

con:
- Slightly more complicated local timing (cells need an extra register to stagger the streams).
- Latency: results appear only after traversing all cells (so not constant-time like W1).

## Remarks
- The space of designs is large
- Once you find one systolic design, you can find many others
    - just need to understand each's trade-offs
- I/O costs matter
- Multiplier–accumulator hardware trade-off
    - Each systolic cell needs a multiplier + adder.
    - A common cost-effective option: combine them into a single multiplier–accumulator (MAC) unit (one multiplication and accumulation per cycle).
    - But if you separate multiplier and adder, you can sometimes overlap their execution and improve throughput.
- Can pipeline operations (fig 10) to improve throughput

### 1. Multiple use of each input
- Key reason systolic arrays overcome I/O bottlenecks.
- Every time you fetch an input from memory, you want to reuse it in many computations before discarding.
- Two ways to achieve this:
    - Global broadcast/fan-in (semi-systolic designs like B1/F).
    - Traveling inputs that visit each cell in turn (pure systolic designs like R1/W1).

### 2. Extensive concurrency
- **Pipelining** the stages of one computation (e.g., B1, where a result is assembled as it moves through).
- **Multiprocessing** multiple computations in parallel (e.g., R1, where several y values are being accumulated simultaneously in different cells).

Many cells working together

### 3. Few simple cell types
- Since systolic arrays may contain hundreds or thousands of cells, the simplicity of each cell is crucial.
- Simpler cells → lower design cost, easier VLSI layout, better modularity.

### 4. Simple, regular data and control flows
- Pure systolic arrays avoid long-distance, irregular communication.
- Data flows only between neighbors, in regular rhythmic patterns.
- Only global signal: the system clock (though asynchronous/local handshakes are possible).
- This regularity makes:
    - Wiring area-efficient.
    - Timing predictable (even if clock skew exists along a 1D array).
    - Design scalable and modular (e.g., extend array length with no changes to control).


### Key algorithms and techniques:

1. **Matrix Multiplication Systolic Array:**
- 2D grid of processors with data flowing in multiple directions (typically north-south and east-west)
- Each PE performs multiply-accumulate operations on streaming matrix elements
- Partial results accumulate as they propagate through the array structure
- Achieves O(n²) processors completing n×n matrix multiplication in O(n) time steps
- Input matrices are fed systematically to maintain continuous processor utilization

2. **Linear Systolic Arrays:**
- One-dimensional chain of processors for computations with linear data dependencies
- Data flows unidirectionally with results propagated to subsequent processors
- Applications: convolution, FIR filtering, polynomial evaluation, string matching
- Simpler control and interconnection compared to 2D arrays
- Trade-off between hardware complexity and computational parallelism

3. **Wavefront Processing:**
- Data dependencies determine the timing of when computations can begin at each PE
- "Wavefront" of computation propagates through the array following dependency constraints
- Each processor begins work only when all required input data is available
- Maximizes parallelism while respecting algorithmic dependencies
- Enables natural load balancing across the processor array

4. **I/O and Data Management:**
- Edge processors handle communication with external memory systems
- Careful data scheduling ensures continuous utilization of internal processors
- Buffering strategies to handle irregularities in external data arrival patterns
- Multiple I/O channels may be required to prevent bandwidth bottlenecks
- Data formatting and distribution to match systolic array timing requirements

## Systolic Design Principles

**Regularity and Modularity:**
- Identical processing elements replicated in regular patterns
- Simple, uniform interconnection structure
- Modular design allows scaling to different problem sizes
- Reduces design complexity and verification effort

**Rhythmic Data Flow:**
- Synchronized data movement following predictable patterns
- Global timing ensures coordinated operation across all processors
- Eliminates need for complex flow control mechanisms
- Enables high-frequency operation with minimal overhead

**Local Communication:**
- Processors communicate only with immediate physical neighbors
- Minimizes wiring complexity and communication latency
- Communication cost scales favorably with array size
- Reduces power consumption compared to global communication

## Interesting findings and insight
- **Performance scaling**: For suitable algorithms, throughput increases linearly with number of processors while maintaining constant per-processor utilization
- **Cost-performance optimization**: Regular structure significantly reduces design and manufacturing costs compared to irregular parallel systems
- **Algorithm transformation**: Many inherently sequential algorithms can be restructured for efficient systolic execution through clever data scheduling
- **VLSI efficiency**: Systolic designs can achieve very high silicon utilization rates due to regular structure and local communication patterns

Compute-bound: number of operation > number of input and output
- can be speed up with systallic archtecture
- eg: matmul

I/O-bound: number of operation <= number of input and output
- can be speed up with better (thus/usually more expensive) component or interleave memory
- eg: element-wise add


# 📚 References
- Kung, H.T. (1982). Why systolic architectures? Computer, 15(1), 37-46.
- Kung, H.T., & Leiserson, C.E. (1978). Systolic arrays (for VLSI). Sparse Matrix Proceedings.
- Foster, M.J., & Kung, H.T. (1980). The design of special-purpose VLSI chips. Computer, 13(1), 26-40.
- Samira Khan & Kung's video on Systolic Array: https://www.youtube.com/watch?v=lTlpJ2Mz4zs&t=2582s

# Background Concepts: Systolic Computing and VLSI Design

## Historical and Contextual Background
1. **VLSI Revolution (Late 1970s-1980s)**:
   - Rapid advancement in integrated circuit technology enabled millions of transistors per chip
   - Traditional architectures couldn't effectively utilize the available silicon area
   - Need for new design methodologies that could exploit massive parallelism
   - Cost considerations became important as custom chip design became feasible

2. **Communication vs. Computation Trade-offs**:
   - Traditional parallel systems suffered from communication bottlenecks as system size increased
   - Recognition that communication cost must scale appropriately with computation capability
   - Local communication patterns became critical for scalable parallel systems
   - Need to balance processor utilization with data movement overhead

3. **Special-Purpose vs. General-Purpose Computing**:
   - General-purpose processors were becoming inefficient for specialized computational tasks
   - Growing recognition that algorithm-specific hardware could provide significant advantages
   - Need for systematic design methodologies for creating special-purpose systems
   - Cost-effectiveness required balancing specialization with design and manufacturing expenses

## Systolic Array Principles

1. **Synchronous, Rhythmic Operation**:
  - All processors operate according to a global clock with synchronized data movement
  - Data flow follows predictable, repeating patterns similar to biological rhythms
  - Eliminates complex asynchronous flow control and scheduling overhead
  - Enables high-frequency operation with minimal coordination complexity

2. **Spatial and Temporal Parallelism**:
  - Spatial parallelism: multiple processors working simultaneously on different data
  - Temporal parallelism: pipelined execution with multiple computation stages active
  - Combination achieves very high throughput for regular computational patterns
  - Efficient utilization of hardware resources through continuous operation

3. **Algorithm-Architecture Matching**:
     - Hardware structure directly reflects the computational pattern of target algorithms
     - Data movement patterns designed to match algorithmic data dependencies
     - Optimal performance achieved when algorithm and architecture are co-designed
     - May require significant algorithm restructuring but provides substantial benefits

## Modern Applications and Legacy:

- **AI Accelerators**: Google TPU, Cerebras WSE, and other neural network processors implement systolic principles
- **Graphics Processing**: Modern GPUs use similar concepts of regular, parallel execution with local communication
- **Digital Signal Processing**: Systolic designs became standard in DSP chips and real-time signal processing
- **High-Performance Computing**: Influenced design of vector processors and parallel supercomputers
- **FPGA Implementations**: Reconfigurable computing systems frequently implement systolic patterns

## Key Insights for Modern Computing:
- **Specialization Benefits**: Domain-specific architectures can achieve orders-of-magnitude improvements over general-purpose systems
- **Communication Locality**: Keeping communication local and regular is essential for scalable parallel systems
- **Design Regularity**: Simple, repetitive structures reduce design complexity and improve manufacturability
- **Algorithm Co-design**: Hardware and software must be designed together to achieve optimal results

## Useful Resources:
- Original Kung paper: http://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf
- "Introduction to VLSI Systems" by Mead & Conway for VLSI design context
- Google TPU papers for modern systolic array implementations in neural network acceleration
- "Parallel Computer Architecture" by Culler, Singh & Gupta for broader parallel computing context

# SIMD

# Dataflow architectures