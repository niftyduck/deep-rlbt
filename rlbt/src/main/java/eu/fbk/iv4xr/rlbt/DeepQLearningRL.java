package eu.fbk.iv4xr.rlbt;

import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.learningrate.LearningRate;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionUtils;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import eu.fbk.iv4xr.rlbt.labrecruits.LabRecruitsState;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import burlap.mdp.core.oo.state.ObjectInstance;
import eu.fbk.iv4xr.rlbt.labrecruits.LabRecruitsEntityObject;
import world.LabEntity;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;


/**
 * Deep Q-Network (DQN) implementation for LabRecruits.
 *
 * Unlike tabular Q-learning, there is no Q-table (no HashMap of states/nodes).
 * The neural network replaces the entire table: given a state encoded as a feature
 * vector, it returns Q-values for all actions in a single forward pass.
 *
 * This implementation includes the two core DQN stabilisation techniques:
 *
 * 1. Experience Replay (replay buffer): past transitions (s, a, r, s', done) are
 *    stored in a circular buffer. At each step, a random mini-batch is sampled from
 *    it for training. This breaks the temporal correlation between consecutive samples
 *    and allows each transition to be reused multiple times.
 *
 * 2. Target Network: a frozen copy of the main network used exclusively to compute
 *    the Bellman target. It is re-synchronized with the main network every
 *    TARGET_UPDATE_FREQUENCY steps. Without it, the target shifts at every gradient
 *    step, causing the network to chase a moving target.
 */
public class DeepQLearningRL extends MDPSolver implements QProvider, LearningAgent {

    /** Main network: updated at every training step via mini-batch gradient descent. */
    private MultiLayerNetwork network;

    /**
     * Target network: a frozen copy of the main network used to compute stable
     * Bellman targets. Re-synchronized every TARGET_UPDATE_FREQUENCY global steps.
     */
    private MultiLayerNetwork targetNetwork;

    /**
     * Fixed list of all entity IDs in the level. It determines both the input size
     * (one feature per entity) and the output size (one Q-value per action/entity)
     * of the network.
     */
    private List<String> entityIds;

    /** Current epsilon for the epsilon-greedy policy. Decays toward epsilonMin over episodes. */
    private double epsilongr;
    /** Floor value for epsilon decay: exploration never drops below this threshold. */
    private double epsilonMin;
    private double decayedEpsilonstep;
    protected LearningRate learningRate;
    protected EpsilonGreedy learningPolicy;
    protected int maxEpisodeSize;
    /** Number of neurons in each hidden layer of the network. */
    private int hiddenSize;

    /** Step counter within the current episode. Reset at the start of each episode. */
    protected int eStepCounter;

    /**
     * Total number of steps taken across all episodes.
     * Used to schedule target network synchronisation.
     */
    protected int totalNumberOfSteps = 0;

    // ---- Replay buffer constants ----

    /** Maximum number of transitions stored in the replay buffer (FIFO, oldest discarded). */
    private static final int REPLAY_BUFFER_CAPACITY = 10000;

    /** Number of transitions sampled per training step. */
    private static final int BATCH_SIZE = 32;

    /**
     * Minimum number of transitions that must be in the buffer before training begins.
     * This warm-up phase ensures the first batches are sufficiently diverse.
     */
    private static final int MIN_REPLAY_SIZE = 64;

    // ---- Target network constants ----

    /**
     * Number of global steps between each copy of the main network weights into the
     * target network. A lower value means faster adaptation but less stability.
     */
    private static final int TARGET_UPDATE_FREQUENCY = 100;

    /** The replay buffer: a circular list of past transitions (s, a, r, s', done). */
    private final List<Transition> replayBuffer = new ArrayList<>();

    /** Shared RNG for uniform random sampling from the replay buffer. */
    private final Random random = new Random();

    // ---- Getters ----
    public double getEpsilongr() { return epsilongr; }
    public double getEpsilonMin() { return epsilonMin; }
    public double getDecayedEpsilonstep() { return decayedEpsilonstep; }
    public LearningRate getLearningRate() { return learningRate; }
    public EpsilonGreedy getLearningPolicy() { return learningPolicy; }
    public int getMaxEpisodeSize() { return maxEpisodeSize; }
    public int getHiddenSize() { return hiddenSize; }
    public int getLastNumSteps() { return eStepCounter; }
    public int getTotalNumberOfSteps() { return totalNumberOfSteps; }

    // ---- Setters ----
    public void setEpsilongr(double epsilongr) { this.epsilongr = epsilongr; }
    public void setEpsilonMin(double epsilonMin) { this.epsilonMin = epsilonMin; }
    public void setDecayedEpsilonStep(double decayedEpsilonStep) { this.decayedEpsilonstep = decayedEpsilonStep; }
    public void setLearningRate(LearningRate learningRate) { this.learningRate = learningRate; }
    public void setLearningPolicy(EpsilonGreedy learningPolicy) { this.learningPolicy = learningPolicy; }
    public void setMaxEpisodeSize(int maxEpisodeSize) { this.maxEpisodeSize = maxEpisodeSize; }
    public void setTotalNumberOfSteps(int totalNumberOfSteps) { this.totalNumberOfSteps = totalNumberOfSteps; }

    /**
     * Constructs the DQN agent. Both the main network and the target network are
     * created with the same architecture; the target network is immediately
     * synchronised with the main network so they start identical.
     *
     * @param domain           the domain in which to learn
     * @param gamma            discount factor γ for future rewards (e.g. 0.99)
     * @param entityIds        fixed list of all entity IDs in the level
     * @param learningRate     Adam optimizer learning rate (e.g. 0.001)
     * @param epsilon          initial exploration rate (1.0 = full random, 0.0 = greedy)
     * @param decayEpsilonStep amount subtracted from epsilon at the end of each episode
     * @param maxEpisodeSize   maximum steps per episode before forced termination
     * @param epsilonMin       floor value for epsilon: exploration never drops below this
     * @param hiddenSize       number of neurons per hidden layer (both layers use the same size)
     */
    public DeepQLearningRL(SADomain domain, double gamma,
                           List<String> entityIds,
                           double learningRate, double epsilon,
                           double decayEpsilonStep, int maxEpisodeSize,
                           double epsilonMin, int hiddenSize) {

        /*
         * null: no HashableStateFactory needed — the neural network replaces the Q-table
         * entirely, taking a feature vector as input and returning Q-values as output.
         */
        this.solverInit(domain, gamma, null);
        this.entityIds = entityIds;
        int size = entityIds.size();
        this.epsilongr = epsilon;
        this.epsilonMin = epsilonMin;
        this.decayedEpsilonstep = decayEpsilonStep;
        this.maxEpisodeSize = maxEpisodeSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = new ConstantLR(learningRate);
        this.learningPolicy = new EpsilonGreedy(this, epsilon);

        this.network = buildNetwork(size, size, learningRate, hiddenSize);

        // Target network starts as an exact copy of the main network
        this.targetNetwork = buildNetwork(size, size, learningRate, hiddenSize);
        updateTargetNetwork();
    }

    /**
     * Builds a fully-connected neural network with two hidden layers (64 units, ReLU)
     * and a linear output layer (IDENTITY activation) to allow unbounded Q-values.
     * Adam is used as the optimizer.
     *
     * @param inputSize  number of input features (= |entityIds|)
     * @param outputSize number of output Q-values (= |entityIds|)
     * @param lr         Adam learning rate
     * @param hiddenSize number of neurons per hidden layer (configurable via burlap.network.hidden_size)
     * @return the initialised MultiLayerNetwork (Xavier weight initialisation)
     */
    private MultiLayerNetwork buildNetwork(int inputSize, int outputSize, double lr, int hiddenSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize).nOut(hiddenSize).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder()
                        .nIn(hiddenSize).nOut(hiddenSize).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(hiddenSize).nOut(outputSize).activation(Activation.IDENTITY).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    /**
     * Encodes a LabRecruits state as a binary feature vector of length |entityIds|.
     * Each position i corresponds to entityIds.get(i):
     *   - 1.0 if the entity is active (isOn for switches, isOpen for doors)
     *   - 0.0 if inactive or not yet observed by the agent
     *
     * The result is reshaped to (1, |entityIds|) because DL4J always expects a
     * 2D matrix where rows = batch size and columns = features.
     *
     * @param s the current LabRecruits state
     * @return a (1, n) INDArray suitable as direct network input
     */
    private INDArray encodeState(State s) {
        LabRecruitsState lrs = (LabRecruitsState) s;
        Map<String, ObjectInstance> observedEntities = lrs.getObjectsMap();
        float[] entities = new float[entityIds.size()];

        for (int i = 0; i < entityIds.size(); i++) {
            String id = entityIds.get(i);
            if (!observedEntities.containsKey(id)) {
                entities[i] = 0.0f;
                continue;
            }
            LabRecruitsEntityObject object = (LabRecruitsEntityObject) observedEntities.get(id);
            LabEntity entity = (LabEntity) object.getLabRecruitsEntity();

            if (entity.type.equalsIgnoreCase(LabEntity.DOOR))
                entities[i] = entity.getBooleanProperty("isOpen") ? 1.0f : 0.0f;
            else if (entity.type.equalsIgnoreCase(LabEntity.SWITCH))
                entities[i] = entity.getBooleanProperty("isOn") ? 1.0f : 0.0f;
        }

        return Nd4j.create(entities).reshape(1, entityIds.size());
    }

    /**
     * Returns all Q-values for a given state via a forward pass on the MAIN network.
     * Required by QProvider — called internally by EpsilonGreedy to select actions.
     *
     * @param s the current state
     * @return list of QValue objects, one per applicable action
     */
    @Override
    public List<QValue> qValues(State s) {
        INDArray qVals = network.output(encodeState(s));
        List<Action> actions = ActionUtils.allApplicableActionsForTypes(this.domain.getActionTypes(), s);
        List<QValue> result = new ArrayList<>();
        for (Action a : actions) {
            int idx = entityIds.indexOf(a.actionName());
            double q = (idx >= 0) ? qVals.getDouble(0, idx) : 0.0;
            result.add(new QValue(s, a, q));
        }
        return result;
    }

    /**
     * Returns the Q-value for a specific (state, action) pair via the MAIN network.
     *
     * @param s the state
     * @param a the action
     * @return Q(s, a) according to the current main network
     */
    @Override
    public double qValue(State s, Action a) {
        INDArray qVals = network.output(encodeState(s));
        int idx = entityIds.indexOf(a.actionName());
        return (idx >= 0) ? qVals.getDouble(0, idx) : 0.0;
    }

    /**
     * Returns the maximum Q-value over all actions for a given state.
     * Equivalent to V*(s) under the current learned policy.
     *
     * @param s the state
     * @return max_a Q(s, a)
     */
    @Override
    public double value(State s) {
        return network.output(encodeState(s)).max(1).getDouble(0);
    }

    @Override
    public Episode runLearningEpisode(Environment env) {
        return this.runLearningEpisode(env, -1);
    }

    /**
     * Runs one full training episode using the full DQN algorithm.
     *
     * At each step:
     *   1. Encode the current state as a feature vector.
     *   2. Select an action via epsilon-greedy (main network, via EpsilonGreedy policy).
     *   3. Execute the action in the LabRecruits environment.
     *   4. Encode the resulting next state.
     *   5. Store the transition (s, a, r, s', done) in the replay buffer.
     *   6. Sample a random mini-batch from the buffer and train the main network
     *      (only once MIN_REPLAY_SIZE transitions have been collected).
     *   7. Every TARGET_UPDATE_FREQUENCY steps, copy the main network weights into
     *      the target network.
     *
     * At the end of the episode, epsilon is reduced by decayedEpsilonstep
     * (minimum clamped at 0.1).
     *
     * @param env      the LabRecruits environment
     * @param maxSteps maximum steps before the episode is cut off (-1 = unlimited)
     * @return the Episode record containing all transitions and rewards
     */
    @Override
    public Episode runLearningEpisode(Environment env, int maxSteps) {
        System.out.println("----------DeepQLearningRL : Starting runLearningEpisode()----------------------");
        State curState = env.currentObservation();
        Episode ea = new Episode(curState);
        eStepCounter = 0;

        while (!env.isInTerminalState() && (eStepCounter < maxSteps || maxSteps == -1)) {
            System.out.println("==================DeepQL - Next turn for this episode==================================");

            LabRecruitsState curlabState = (LabRecruitsState) curState;
            if (curlabState.numObjects() == 0) {
                System.out.println(" BUG : Empty Observation of RL active agent. Ending Episode...");
                break;
            }

            // Step 1: Encode current state as a fixed-size feature vector
            INDArray stateVec = encodeState(curState);

            // Step 2: Select action via epsilon-greedy using the main network
            Action action = learningPolicy.action(curState);
            System.out.println("Action Selected : in runLearningEpisode(): " + action.actionName());

            // Step 3: Execute the action in the environment
            EnvironmentOutcome eo = env.executeAction(action);

            // Step 4: Encode the next state
            INDArray nextStateVec = encodeState(eo.op);

            // Step 5: Store transition in the replay buffer (.dup() to prevent mutation)
            int actionIdx = entityIds.indexOf(action.actionName());
            addToReplayBuffer(new Transition(stateVec.dup(), actionIdx, eo.r, nextStateVec.dup(), eo.terminated));

            // Step 6: Train the main network on a random mini-batch from the buffer
            trainOnBatch();

            // Step 7: Periodically synchronise the target network with the main network
            if (totalNumberOfSteps % TARGET_UPDATE_FREQUENCY == 0) {
                updateTargetNetwork();
                System.out.println("Target network updated at step " + totalNumberOfSteps);
            }

            ea.transition(action, eo.op, eo.r);
            curState = eo.op;
            eStepCounter++;
            totalNumberOfSteps++;
        }

        System.out.println("=============Episode summary==========================");
        System.out.println("Action sequence " + ea.actionSequence.size() + "  =" + ea.actionSequence);
        System.out.println("Reward sequence " + ea.rewardSequence.size() + "  =" + ea.rewardSequence);
        System.out.println("Epsilon value = " + this.epsilongr);
        System.out.println("Replay buffer size = " + replayBuffer.size());

        // Decay epsilon at the end of each episode (floor = epsilonMin)
        if (this.epsilongr > this.epsilonMin)
            this.epsilongr = Math.max(this.epsilonMin, this.epsilongr - decayedEpsilonstep);
        this.learningPolicy = new EpsilonGreedy(this, this.epsilongr);
        System.out.println("Decay Epsilon Value : End of an episode = " + this.epsilongr);

        return ea;
    }

    /**
     * Adds a transition to the replay buffer.
     * If the buffer has reached REPLAY_BUFFER_CAPACITY, the oldest entry is removed
     * first (FIFO circular behaviour).
     *
     * @param t the transition to store
     */
    private void addToReplayBuffer(Transition t) {
        if (replayBuffer.size() >= REPLAY_BUFFER_CAPACITY) {
            replayBuffer.remove(0);
        }
        replayBuffer.add(t);
    }

    /**
     * Samples a uniformly random mini-batch from the replay buffer and performs one
     * gradient descent step on the MAIN network.
     *
     * Bellman targets are computed using the TARGET network (not the main network),
     * providing stable training signals:
     *   - terminal transition:     target = r
     *   - non-terminal transition: target = r + γ * max_a'( Q_target(s', a') )
     *
     * All BATCH_SIZE transitions are stacked into matrices and processed in a single
     * network.fit() call, making the update efficient and reducing gradient variance
     * compared to single-sample updates.
     *
     * Does nothing if the buffer has fewer than MIN_REPLAY_SIZE transitions.
     */
    private void trainOnBatch() {
        if (replayBuffer.size() < MIN_REPLAY_SIZE) return;

        // Uniform random sampling (with replacement) from the replay buffer
        List<Transition> batch = new ArrayList<>(BATCH_SIZE);
        for (int i = 0; i < BATCH_SIZE; i++) {
            batch.add(replayBuffer.get(random.nextInt(replayBuffer.size())));
        }

        // Stack individual (1, n) state vectors into (BATCH_SIZE, n) matrices
        List<INDArray> stateList     = new ArrayList<>(BATCH_SIZE);
        List<INDArray> nextStateList = new ArrayList<>(BATCH_SIZE);
        for (Transition t : batch) {
            stateList.add(t.state);
            nextStateList.add(t.nextState);
        }
        INDArray stateBatch     = Nd4j.vstack(stateList);
        INDArray nextStateBatch = Nd4j.vstack(nextStateList);

        // Forward pass on main network  → current Q-values (base for target matrix)
        INDArray currentQBatch = network.output(stateBatch);
        // Forward pass on TARGET network → next-state Q-values (stable Bellman targets)
        INDArray nextQBatch    = targetNetwork.output(nextStateBatch);

        // Build target matrix: copy current Q-values and overwrite only the executed action
        INDArray targetBatch = currentQBatch.dup();
        for (int i = 0; i < batch.size(); i++) {
            Transition t = batch.get(i);
            double target;
            if (t.terminated) {
                target = t.reward;
            } else {
                double maxNextQ = nextQBatch.getRow(i).maxNumber().doubleValue();
                target = t.reward + this.gamma * maxNextQ;
            }
            targetBatch.putScalar(new int[]{i, t.actionIdx}, target);
        }

        // Single gradient descent step on the whole batch
        network.fit(stateBatch, targetBatch);
    }

    /**
     * Copies the current weights of the main network into the target network.
     * The copy is made with .dup() so that the two sets of parameters remain
     * completely independent after the call.
     * Called every TARGET_UPDATE_FREQUENCY global steps during training,
     * and also immediately after deserialising a saved model.
     */
    private void updateTargetNetwork() {
        targetNetwork.setParams(network.params().dup());
    }

    /**
     * Tests the learned policy without modifying any internal state.
     * Differences from runLearningEpisode:
     *   - Action selection is always greedy (no epsilon-greedy randomness).
     *   - network.fit() is never called (no learning).
     *   - The replay buffer, epsilon and step counters are never modified.
     *
     * @param env      the LabRecruits environment (must already be started)
     * @param maxSteps maximum steps per test episode (-1 = unlimited)
     * @return the Episode record with all transitions and rewards
     */
    public Episode testDeepQLearningAgent(Environment env, int maxSteps) {
        System.out.println("---------------------------------------------------------------\n Test DeepQLearning agent");
        State curState = env.currentObservation();
        Episode episode = new Episode(curState);
        int stepCounter = 0;

        while (!env.isInTerminalState() && (stepCounter < maxSteps || maxSteps == -1)) {
            LabRecruitsState curlabState = (LabRecruitsState) curState;
            if (curlabState.numObjects() == 0) {
                System.out.println(" BUG : Empty Observation of RL active agent. Ending Episode...");
                break;
            }

            INDArray qValues = network.output(encodeState(curState));
            Action action = getMaxValuedAction(curState, qValues);
            if (action == null) {
                System.out.println("No action available from state: " + curState.toString());
                break;
            }
            System.out.println("Action selected: " + action.actionName());

            EnvironmentOutcome eo = env.executeAction(action);
            episode.transition(action, eo.op, eo.r);
            curState = eo.op;
            stepCounter++;
        }

        System.out.println("=============Test Episode summary==========================");
        System.out.println("Action sequence " + episode.actionSequence.size() + "  =" + episode.actionSequence);
        System.out.println("Reward sequence " + episode.rewardSequence.size() + "  =" + episode.rewardSequence);

        return episode;
    }

    /**
     * Returns the action with the highest Q-value for the given state.
     * Used exclusively by testDeepQLearningAgent() for greedy action selection.
     *
     * @param s       the current state
     * @param qValues the main network output for this state (pre-computed)
     * @return the best action, or null if no applicable actions exist
     */
    private Action getMaxValuedAction(State s, INDArray qValues) {
        List<Action> actions = ActionUtils.allApplicableActionsForTypes(this.domain.getActionTypes(), s);
        Action best = null;
        double maxQ = Double.NEGATIVE_INFINITY;
        for (Action a : actions) {
            int idx = entityIds.indexOf(a.actionName());
            double q = (idx >= 0) ? qValues.getDouble(0, idx) : 0.0;
            if (q > maxQ) {
                maxQ = q;
                best = a;
            }
        }
        return best;
    }

    /**
     * Prints a human-readable summary of the main network architecture and entity mapping.
     * Replaces the Q-table printout used in tabular Q-learning.
     *
     * @param ps the output stream (System.out or a FileOutputStream)
     */
    public void printNetworkSummary(PrintStream ps) {
        ps.println("\n\n=====================Deep Q-Network Summary========================================");
        ps.println("Entity IDs (actions): " + entityIds);
        ps.println("Number of entities / actions: " + entityIds.size());
        ps.println("Network layers: " + network.getnLayers());
        ps.println(network.summary());
        ps.println("Replay buffer size: " + replayBuffer.size());
        ps.println("Total training steps: " + totalNumberOfSteps);
        ps.println("----------------------------------------------------------------------------");
    }

    /**
     * Saves the main network weights, architecture and Adam updater state to disk
     * using DL4J's ModelSerializer. Saving the updater state allows training to
     * resume without resetting Adam's moment estimates.
     *
     * @param path destination file path (e.g. "rlbt-files/results/qnetwork.ser")
     */
    public void serializeModel(String path) {
        try {
            ModelSerializer.writeModel(network, new File(path), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads the main network weights and architecture from disk.
     * After loading, the target network is immediately re-synchronised with the
     * restored weights so that both networks are consistent.
     *
     * @param path source file path of the previously saved model
     */
    public void deserializeModel(String path) {
        try {
            this.network = ModelSerializer.restoreMultiLayerNetwork(new File(path));
            updateTargetNetwork();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Fully resets the agent: re-initialises both network weight sets (Xavier),
     * clears the replay buffer, and resets all step counters.
     * Epsilon is NOT reset here — use setEpsilongr() separately if needed.
     */
    @Override
    public void resetSolver() {
        this.network.init();
        updateTargetNetwork();
        this.replayBuffer.clear();
        this.eStepCounter = 0;
        this.totalNumberOfSteps = 0;
    }

    /**
     * Immutable record of a single environment transition stored in the replay buffer.
     *
     * @param state      encoded state vector (1, n) at time t — stored as a copy
     * @param actionIdx  index of the executed action within entityIds
     * @param reward     immediate reward r received after executing the action
     * @param nextState  encoded state vector (1, n) at time t+1 — stored as a copy
     * @param terminated true if the episode ended after this transition
     */
    private static class Transition {
        final INDArray state;
        final int actionIdx;
        final double reward;
        final INDArray nextState;
        final boolean terminated;

        Transition(INDArray state, int actionIdx, double reward, INDArray nextState, boolean terminated) {
            this.state = state;
            this.actionIdx = actionIdx;
            this.reward = reward;
            this.nextState = nextState;
            this.terminated = terminated;
        }
    }
}