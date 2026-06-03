package eu.fbk.iv4xr.rlbt;

import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.learningrate.LearningRate;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearningStateNode;
import burlap.behavior.singleagent.options.EnvironmentOptionOutcome;
import burlap.behavior.singleagent.options.Option;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.debugtools.DPrint;
import burlap.debugtools.RandomFactory;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import eu.fbk.iv4xr.rlbt.labrecruits.LabRecruitsState;
import eu.fbk.iv4xr.rlbt.labrecruits.distance.JaccardDistance;
import eu.fbk.iv4xr.rlbt.utils.SerializationUtil;
import eu.fbk.iv4xr.rlbt.utils.Utils;
import eu.fbk.iv4xr.rlbt.distance.StateDistance;

import javax.management.RuntimeErrorException;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


public class DeepQLearningRL extends MDPSolver implements QProvider, LearningAgent, Planner {

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
     */
    private int numActions;

    /**
     * The size of the state vector is equal to the number of entities,
     * since we encode the state as a vector of features (isOn, isOpen) for each entity.
     */
    private int stateVectorSize;


    /**
     * Same parameters as QLearningRL.java
     */
    private double epsilongr;
    private double decayedEpsilonstep;
    protected LearningRate learningRate;
    protected Policy learningPolicy;
    protected int maxEpisodeSize;
    // Manca roba da QLearningRL?


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
        net.init();   // ← qui i pesi vengono inizializzati (Xavier automatico)
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
        Map<String, ObjectInstance> observed = lrs.getObjectsMap(); // all observed entities

        float[] vec = new float[entityIds.size()]; // non-observed entities will be 0.0

        for (int i = 0; i < entityIds.size(); i++) {
            String id = entityIds.get(i);

            // first step: if the entities is not observed, then it is automatically set to 0
            if (!observed.containsKey(id)) {
                vec[i] = 0.0f;
                continue;
            }

            LabRecruitsEntityObject obj = (LabRecruitsEntityObject) observed.get(id);
            LabEntity entity = (LabEntity) obj.getLabRecruitsEntity();

            if (entity.type.equalsIgnoreCase(LabEntity.DOOR)) {
                vec[i] = entity.getBooleanProperty("isOpen") ? 1.0f : 0.0f;
            } else if (entity.type.equalsIgnoreCase(LabEntity.SWITCH)) {
                vec[i] = entity.getBooleanProperty("isOn") ? 1.0f : 0.0f;
            }
        }

        // builds a tensor DL4J from the float[].
        // the reshape is needed because the network expects a 2D tensor (batch size, features),
        // and here we have only one state (batch size = 1)
        return Nd4j.create(vec).reshape(1, entityIds.size());

        /**
         * Note: a batch is a group of examples that we pass to the network all together in a single call,
         * instead of one by one. DL4J always uses a batch because Neural Networks are optimized to work
         * on matrices, and therefore represent the input as a 2D matrix:
         * -> rows      = number of examples in the batch
         * -> columns   = number of features (size of the state vector)
         */
    }
}

