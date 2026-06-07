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


/**
 * The behavior here is slightly different. We don't use nodes (i.e. a row of Q-table, memorized as HashMap<HashableState, QLearningStateNode>).
 * Here nodes don't exist: the neural network substitutes the entire map: given an state (encoded as a vector) in input, the network
 * returns the in output directly all Q-values for each action in a single forward pass. No lookup table, no nodes, no HashMap.
 * Moreover, StateDistance (JaccardDistance) is not needed. In normal QLearning, it's useful to find similar states, but the neural network
 * naturally generalizes similar states (advantage wrt tabular).
 */

public class DeepQLearningRL extends MDPSolver implements QProvider, LearningAgent {

    /**
     * Instead of a qFunction, here, we use a neural network
     */
    private MultiLayerNetwork network;


    /**
     * This contains all entity ids of the level, since the network has fixed size
     */
    private List<String> entityIds; // fixed level list

    /**
     * In QLearningRL.java, the number of actions was dynamic. If the agent saw 3 entities,
     * it had 3 actions. Now we have a fixed number of actions that depends on the
     * number of entities of the whole level. This is the only drawback of neural network:
     * you need to give it a fixed number of layers.
     * (numActions is not stored as a field -- entityIds.size() is used directly)
     */


    /**
     * Same parameters as QLearningRL.java
     */
    private double epsilongr;
    private double decayedEpsilonstep;
    protected LearningRate learningRate;
    protected Policy learningPolicy;
    protected int maxEpisodeSize;

    /**
     * A counter for counting the number of steps in an episode that have been taken thus far
     */
    protected int eStepCounter;

    /**
     * The total number of learning steps performed by this agent.
     */
    protected int totalNumberOfSteps = 0;


    public double getEpsilongr() { return epsilongr; }
    public double getDecayedEpsilonstep() { return decayedEpsilonstep; }
    public LearningRate getLearningRate() { return learningRate; }
    public Policy getLearningPolicy() { return learningPolicy; }
    public int getMaxEpisodeSize() { return maxEpisodeSize; }
    public int getLastNumSteps(){ return eStepCounter; }
    public int getTotalNumberOfSteps() { return totalNumberOfSteps; }

    public void setEpsilongr(double epsilongr) { this.epsilongr = epsilongr; }
    public void setDecayedEpsilonStep(double decayedEpsilonStep) { this.decayedEpsilonstep = decayedEpsilonStep; }
    public void setLearningRate(LearningRate learningRate) { this.learningRate = learningRate; }
    public void setLearningPolicy(Policy learningPolicy) { this.learningPolicy = learningPolicy; }
    public void setMaxEpisodeSize(int maxEpisodeSize) { this.maxEpisodeSize = maxEpisodeSize; }
    public void setTotalNumberOfSteps(int totalNumberOfSteps) { this.totalNumberOfSteps = totalNumberOfSteps; }

    /**
     * Initializes the network
     *  Constructor
     * @param domain the domain in which to learn
     * @param gamma the discount factor
     * @param entityIds the list of all entities
     * @param learningRate the learning rate
     * @param epsilon the initial epsilon for the epsilon-greedy policy
     * @param decayEpsilonStep the amount by which to decay epsilon after each episode
     * @param maxEpisodeSize the maximum number of steps per episode
     */
    public DeepQLearningRL(SADomain domain, double gamma,
                           List<String> entityIds, // fixed level list
                           double learningRate, double epsilon,
                           double decayEpsilonStep, int maxEpisodeSize) {

        /* null perché non usiamo una HashableStateFactory, ma una rete neurale
         che prende in input un vettore di features (stato) e restituisce un vettore
          di Q-values (azioni) */
        this.solverInit(domain, gamma, null);
        this.entityIds = entityIds;
        int size = entityIds.size();
        this.epsilongr = epsilon;
        this.decayedEpsilonstep = decayEpsilonStep;
        this.maxEpisodeSize = maxEpisodeSize;
        this.learningRate = new ConstantLR(learningRate);
        this.learningPolicy = new EpsilonGreedy(this, epsilon);

        this.network = buildNetwork(size, size, learningRate);
    }

    /**
     *  Build the Neural Network
     * @param inputSize
     * @param outputSize
     * @param lr the learning rate
     * @return the actual network
     */
    private MultiLayerNetwork buildNetwork(int inputSize, int outputSize, double lr) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize).nOut(64).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder()
                        .nIn(64).nOut(64).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(64).nOut(outputSize).activation(Activation.IDENTITY).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();   // automatic Xavier initializes the weights
        return net;
    }


    /**
     * Encode the state as a vector of features (isOn, isOpen) for each entity.
     * The order of entities is fixed based on the entityIds list, so that the
     * same state always produces the same vector.
     * @param s
     * @return
     */
    private INDArray encodeState(State s) {
        LabRecruitsState lrs = (LabRecruitsState) s;

        Map<String, ObjectInstance> observedEntities = lrs.getObjectsMap(); // all observed entities
        float[] entities = new float[entityIds.size()];                     // all entities

        for (int i = 0; i < entityIds.size(); i++) {
            String id = entityIds.get(i);

            // if the entities is not observed, then it is automatically set to 0.0
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

        // builds a tensor DL4J from the float[].
        // the reshape is needed because the network expects a 2D tensor (batch size, features),
        // and here we have only one state (batch size = 1)
        return Nd4j.create(entities).reshape(1, entityIds.size());

        /**
         * Note: a batch is a group of examples that we pass to the network all together in a single call,
         * instead of one by one. DL4J always uses a batch because Neural Networks are optimized to work
         * on matrices, and therefore represent the input as a 2D matrix:
         * -> rows      = number of examples in the batch
         * -> columns   = number of features (size of the state vector)
         */
    }

    /**
     * Returns all Q-values for a given state by doing a forward pass through the network.
     * Required by QProvider -- used internally by EpsilonGreedy to choose actions.
     * @param s
     * @return
     */
    @Override
    public List<QValue> qValues(State s) {
        INDArray qVals = network.output(encodeState(s));
        List<Action> actions = ActionUtils.allApplicableActionsForTypes(this.domain.getActionTypes(), s);
        List<QValue> result = new ArrayList<>();
        for (Action a : actions) {
            int idx = entityIds.indexOf(a.actionName());

            // if the action is not in entityIds, its Q-value defaults to 0.0
            double q = (idx >= 0) ? qVals.getDouble(0, idx) : 0.0;
            result.add(new QValue(s, a, q));
        }
        return result;
    }

    /**
     * Returns the Q-value for a specific (state, action) pair.
     * @param s
     * @param a
     * @return
     */
    @Override
    public double qValue(State s, Action a) {
        INDArray qVals = network.output(encodeState(s));
        int idx = entityIds.indexOf(a.actionName());
        return (idx >= 0) ? qVals.getDouble(0, idx) : 0.0;
    }

    /**
     * Returns the maximum Q-value over all actions for a given state.
     * @param s
     * @return
     */
    @Override
    public double value(State s) {
        // max(1) computes the max over columns (axis 1), returning a (1,1) tensor
        return network.output(encodeState(s)).max(1).getDouble(0);
    }

    @Override
    public Episode runLearningEpisode(Environment env) {
        return this.runLearningEpisode(env, -1);
    }

    @Override
    public Episode runLearningEpisode(Environment env, int maxSteps) {
        System.out.println("----------DeepQLearningRL : Starting runLearningEpisode()----------------------");
        State curState = env.currentObservation();
        Episode ea = new Episode(curState);

        eStepCounter = 0;

        while (!env.isInTerminalState() && (eStepCounter < maxSteps || maxSteps == -1)) {
            System.out.println("==================DeepQL - Next turn for this episode==================================");

            // check for empty observation (same guard as QLearningRL)
            LabRecruitsState curlabState = (LabRecruitsState) curState;
            if (curlabState.numObjects() == 0) {
                System.out.println(" BUG : Empty Observation of RL active agent. Ending Episode...");
                break;
            }

            /**
             * Step 1: Encoding of the state. The state is the set of all entities and their state (isOpen/isOn).
             * Transform (isOpen) and (isOn) to 1.0 and (!isOpen) and (!isOn) to 0.0
             */
            INDArray stateVec = encodeState(curState); // (1, n) vector


            /**
             * Step 2: Forward pass. The network receives the state vector and outputs
             * a set of Q-values (one for each action).
             *
             * Example:
             * [switch1, switch2, switch3]
             * [  0.3  ,   0.7  ,   0.1  ] -> going through switch2 is the best option for now
             */
            INDArray qValues = network.output(stateVec); // forward pass, return (1,n ) Q-values for all actions


            /**
             * Step 3: Choose an action to do based on the qValues.
             * learningPolicy.action(s) calls EpsilonGreedy(), which
             * calls internally qValues(curState) (that does another forward pass),
             * and either chooses the action with the highest Q-value or random.
             */
            Action action = learningPolicy.action(curState); // choose action to do via epsilon-greedy
            System.out.println("Action Selected : in runLearningEpisode(): " + action.actionName());


            /**
             * Step 4: Execution of the actual action.
             * This connects to APlib and executes the action in LabRecruits. It returns
             * the new state eo.ep, the reward eo.r and eo.terminated (1 if terminated)
             */
            EnvironmentOutcome eo = env.executeAction(action);

            /**
             * Step 5: Encoding of the next state.
             * It's the same identical process but for the next state: encoding + forward pass.
             * It's useful to calculate how worth it is to go to s'.
             */
            INDArray nextStateVec = encodeState(eo.op); // encode next state
            INDArray nextQValues = network.output(nextStateVec); // compute its Q-value

            /**
             * Step 6: Bellman equation for target.
             * It's the kernel of Q-learning. The question is: "which is the correct value of Q(s, a)?"
             * The answer is the Bellman equation:
             * Q(s, a) = r + γ * max(Q(s', a'))
             *  - r = reward obtained immediately
             *  - γ * max(Q(s', a')) = future attended reward
             *  - If the state is terminal (goal reached), the future is equal to 0: Q(s, a) = r
             *
             *  This target is the correct response that we want from the network for this couple (s,a)
             */
            double target;    // Bellman target: r + gamma * max_a'(Q(s', a'))  -- or just r if terminal
            if (eo.terminated)
                target = eo.r;
            else {
                double maxNextQ = nextQValues.max(1).getDouble(0);
                target = eo.r + this.gamma * maxNextQ;
            }

            /**
             * Step 7: Construction of the target vector.
             * The network produces Q-values for ALL actons, but we know the correct value only for
             * the action that we've just executed. So:
             * - We copy the current network output (so that the error on other actions is zero)
             * - We substitute only the Q-value of the executed action with the target
             * Result: targetVec is equal to the output of the network, except for one position.
             */
            INDArray targetVec = qValues.dup();
            int actionIdx = entityIds.indexOf(action.actionName());
            targetVec.putScalar(new int[]{0, actionIdx}, target);

            /**
             * Step 8: Backpropagation.
             * The network compares its output with targetVec, calculates the error (MSE)
             * and adjusts the weights to reduce it. Only the Q-value of the executed action is updated,
             * the others remain almost the same.
             * Example: The network has produced [0.3, 0.7, 0.1], but the correct target is [1.72, 0.7, 0.1].
             * This is an error. So go back and change the weights in order to correct the final Q-value.
             */
            network.fit(stateVec, targetVec);

            /**
             * Step 9: Advancement.
             * We register the transition of the episode and pass to a new state
             */
            ea.transition(action, eo.op, eo.r);
            curState = eo.op;
            eStepCounter++;
            totalNumberOfSteps++;
        }

        System.out.println("=============Episode summary==========================");
        System.out.println("Action sequence " + ea.actionSequence.size() + "  =" + ea.actionSequence);
        System.out.println("Reward sequence " + ea.rewardSequence.size() + "  =" + ea.rewardSequence);
        System.out.println("Epsilon value = " + this.epsilongr);

        /**
         * Last step: reduce epsilon gradually
         */
        if (this.epsilongr > 0.1)
            this.epsilongr = this.epsilongr - decayedEpsilonstep;
        this.learningPolicy = new EpsilonGreedy(this, this.epsilongr);
        System.out.println("Decay Epsilon Value : End of an episode = " + this.epsilongr);

        return ea;
    }

    /**
     * Tests the learned policy without updating the network.
     * Unlike runLearningEpisode, this method:
     * - always picks the action with the highest Q-value (greedy, no exploration)
     * - never calls network.fit() (no learning)
     * - never decays epsilon or modifies the agent state
     * @param env
     * @param maxSteps
     * @return
     */
    public Episode testDeepQLearningAgent(Environment env, int maxSteps) {
        System.out.println("---------------------------------------------------------------\n Test DeepQLearning agent");
        State curState = env.currentObservation();
        Episode episode = new Episode(curState);

        int stepCounter = 0;

        while (!env.isInTerminalState() && (stepCounter < maxSteps || maxSteps == -1)) {

            // check for empty observation
            LabRecruitsState curlabState = (LabRecruitsState) curState;
            if (curlabState.numObjects() == 0) {
                System.out.println(" BUG : Empty Observation of RL active agent. Ending Episode...");
                break;
            }

            // forward pass: get Q-values for all actions
            INDArray qValues = network.output(encodeState(curState));

            // greedy action selection: pick the action with the highest Q-value (no randomness)
            Action action = getMaxValuedAction(curState, qValues);
            if (action == null) {
                System.out.println("No action available from state: " + curState.toString());
                break;
            }
            System.out.println("Action selected: " + action.actionName());

            // execute action — no network.fit(), no learning
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
     * Used by testDeepQLearningAgent to always pick the best known action.
     * @param s
     * @param qValues the network output for this state (already computed)
     * @return
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
     * Prints a summary of the network in place of a Q-table.
     */
    public void printNetworkSummary(PrintStream ps) {
        ps.println("\n\n=====================Deep Q-Network Summary========================================");
        ps.println("Entity IDs (actions): " + entityIds);
        ps.println("Number of entities / actions: " + entityIds.size());
        ps.println("Network layers: " + network.getnLayers());
        ps.println(network.summary());
        ps.println("----------------------------------------------------------------------------");
    }

    /**
     * Saves the network weights and architecture to disk.
     */
    public void serializeModel(String path) {
        try {
            ModelSerializer.writeModel(network, new File(path), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads the network weights and architecture from disk.
     */
    public void deserializeModel(String path) {
        try {
            this.network = ModelSerializer.restoreMultiLayerNetwork(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void resetSolver() {
        // re-initializes the network weights with Xavier (same as constructor)
        this.network.init();
        this.eStepCounter = 0;
        this.totalNumberOfSteps = 0;
    }


    /**
     * Same signature as QLearningRL.printFinalQtable() for drop-in compatibility.
     */
    public void printFinalQtable(PrintStream ps) {
        printNetworkSummary(ps);
    }

    /**
     * Same signature as QLearningRL.serializeQTable() for drop-in compatibility.
     */
    public void serializeQTable(String path) {
        serializeModel(path);
    }

    /**
     * Same signature as QLearningRL.deserializeQTable() for drop-in compatibility.
     */
    public void deserializeQTable(String path) {
        deserializeModel(path);
    }
}