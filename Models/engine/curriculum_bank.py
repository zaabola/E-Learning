"""
Multi-Variant 5-Concept Knowledge Graph Curriculum Bank for KnowledgeQuest
--------------------------------------------------------------------------
Follows the 5-node Prerequisite DAG (`PREREQ_GRAPH`) used by the RL Game Master:
  - Concept 0 (Foundation A)  ---\
                                  +--> Concept 2 (Route C) ---\
  - Concept 1 (Foundation B)  ---\                             +--> Concept 4 (Boss Synthesis E)
                                  +--> Concept 3 (Route D) ---/

Each difficulty tier holds multiple distinct rotating question variants so
consecutive turns never repeat the exact same prompt. Also accepts custom
single-dict or list-of-dict JSON payloads from Objective 1 ("Notes to Knowledge").
"""

import json
import os

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_ENGINE_DIR))
VISION_CURRICULUM_JSON = os.path.join(
    _REPO_ROOT, 'course_material', 'vision_yolo_ocr', 'course_curriculum.json'
)

PREREQ_GRAPH = {
    0: [],
    1: [],
    2: [0],
    3: [1],
    4: [2, 3],
}

DEFAULT_CURRICULUM = [
    # =========================================================================
    # CONCEPT 0: Neurons & Non-Linear Activations (Foundation A)
    # =========================================================================
    {
        'concept_id': 0,
        'title': 'Neurons & Non-Linear Activations',
        'short_title': 'Activations',
        'prereqs': [],
        'summary_bullets': [
            'Without non-linear activations (ReLU, GELU, SiLU), stacked linear layers collapse into a single linear matrix.',
            'ReLU outputs max(0, x); always-negative pre-activations produce zero gradient ("dying ReLU").',
            'Smooth activations like GELU and SiLU preserve small negative gradients near zero for Transformers and CNNs.',
        ],
        'podcast_hint': (
            '[Podcast Clip 01:12 - 01:27] "Without a non-linear activation like ReLU or GELU '
            'between layers, stacking ten linear matrix multiplications mathematically collapses '
            'into just one single linear layer!"'
        ),
        'easy_mc': [
            {
                'question': 'Why do deep neural networks apply non-linear activation functions like ReLU between layers?',
                'choices': [
                    'To prevent stacked linear layers from collapsing into a single linear transformation',
                    'To force every weight in the network to equal zero',
                    'To delete the input features before training begins',
                    'To replace gradient descent with random guessing',
                ],
                'correct_idx': 0,
                'reference': 'Non-linear activations like ReLU allow stacked layers to learn complex non-linear patterns instead of collapsing into a single linear function.',
            },
            {
                'question': 'What is the output of the ReLU activation function f(x) = max(0, x) when given x = -4.5?',
                'choices': [
                    '-4.5',
                    '0.0',
                    '+4.5',
                    '1.0',
                ],
                'correct_idx': 1,
                'reference': 'ReLU outputs max(0, x), so any negative input produces 0.0 while positive inputs pass through unchanged.',
            },
            {
                'question': 'Which activation function squashes any real-valued logit into a bounded probability between 0 and 1 for binary classification?',
                'choices': [
                    'Sigmoid: 1 / (1 + exp(-x))',
                    'Identity: f(x) = x',
                    'Unbounded ReLU: max(0, x)',
                    'Matrix transpose',
                ],
                'correct_idx': 0,
                'reference': 'The sigmoid function maps real-valued logits smoothly into the bounded interval (0, 1) to represent binary probabilities.',
            },
        ],
        'medium_mc': [
            {
                'question': 'What happens to a standard ReLU neuron f(x) = max(0, x) when its pre-activation input x is strictly negative for all training examples?',
                'choices': [
                    'Its gradient becomes 100 and weights explode immediately',
                    'It outputs 0 with zero local gradient, causing the "dying ReLU" phenomenon',
                    'It automatically converts into a softmax probability distribution',
                    'It doubles the learning rate of the optimizer',
                ],
                'correct_idx': 1,
                'reference': 'When x is always negative, ReLU outputs zero and its derivative is zero, stopping weight updates in a dying ReLU neuron.',
            },
            {
                'question': 'Why do modern Transformers (like DeBERTa and GPT) prefer GELU or SiLU over hard ReLU in feed-forward blocks?',
                'choices': [
                    'They provide smooth non-zero gradients near zero instead of a sharp zero-gradient cutoff',
                    'They remove all trainable parameters from the network',
                    'They only accept integer inputs',
                    'They disable backpropagation in attention layers',
                ],
                'correct_idx': 0,
                'reference': 'Smooth activations like GELU and SiLU preserve small negative gradients near zero, avoiding dead neurons and improving gradient flow.',
            },
            {
                'question': 'If a 4-layer network uses ONLY f(x) = x (no activation functions), what class of decision boundaries can it represent?',
                'choices': [
                    'Arbitrarily curved spirals and concentric rings',
                    'Only a single hyperplane / linear transformation equivalent to a 1-layer linear model',
                    'Decision trees with axis-aligned splits',
                    'Non-parametric nearest-neighbor Voronoi cells',
                ],
                'correct_idx': 1,
                'reference': 'A composition of linear matrix multiplications without non-linear activations reduces to a single linear matrix transformation.',
            },
        ],
        'boss_open': [
            {
                'question': 'Explain in your own words why stacking multiple linear layers without any activation function cannot model non-linear decision boundaries.',
                'reference': 'Multiplying several linear transformations together still produces a single linear transformation, so non-linear activation functions are required to learn curved or complex boundaries.',
            },
            {
                'question': 'What causes the "dying ReLU" problem during training, and how do Leaky ReLU or GELU fix it?',
                'reference': 'A standard ReLU outputs zero and has zero gradient for all negative inputs so its weights stop updating, whereas Leaky ReLU and GELU keep a small non-zero slope for negative values so gradients still flow.',
            },
        ],
    },

    # =========================================================================
    # CONCEPT 1: Loss Functions & Gradient Descent (Foundation B)
    # =========================================================================
    {
        'concept_id': 1,
        'title': 'Loss Functions & Gradient Descent',
        'prereqs': [],
        'podcast_hint': (
            '[Podcast Clip 03:40 - 03:55] "Think of the loss surface as a foggy mountain valley: '
            'the gradient points uphill toward steeper error, so gradient descent steps in the '
            'negative gradient direction scaled by the learning rate."'
        ),
        'easy_mc': [
            {
                'question': 'In gradient descent optimization, which direction are model weights updated relative to the loss gradient?',
                'choices': [
                    'In the opposite (negative) direction of the gradient to minimize loss',
                    'In the positive direction of the gradient to maximize error',
                    'Perpendicular to the dataset size without using derivatives',
                    'Only alphabetically by parameter name',
                ],
                'correct_idx': 0,
                'reference': 'Gradient descent updates weights in the negative direction of the loss gradient to reduce prediction error.',
            },
            {
                'question': 'Which loss function is standard for multi-class classification over mutually exclusive categories?',
                'choices': [
                    'Cross-Entropy Loss over softmax class probabilities',
                    'Counting total characters in the class names',
                    'Adding all input pixel values together',
                    'Cosine of the epoch number',
                ],
                'correct_idx': 0,
                'reference': 'Cross-entropy loss measures divergence between predicted softmax probabilities and the true class label.',
            },
            {
                'question': 'What is the main advantage of Mini-Batch Stochastic Gradient Descent (SGD) over full-dataset batch gradient descent?',
                'choices': [
                    'It updates weights much more frequently using small GPU-friendly batches instead of scanning the entire dataset per step',
                    'It never computes any gradients',
                    'It guarantees zero loss after a single sample',
                    'It eliminates the need for a loss function',
                ],
                'correct_idx': 0,
                'reference': 'Mini-batch gradient descent estimates gradients efficiently on small batches, enabling fast frequent updates that fit in GPU memory.',
            },
        ],
        'medium_mc': [
            {
                'question': 'What typically occurs during training if the learning rate is set several orders of magnitude too high?',
                'choices': [
                    'Training converges cleanly in zero epochs with perfect accuracy',
                    'Weight updates overshoot valleys, causing the loss to oscillate wildly or diverge',
                    'The model parameter count shrinks by half each batch',
                    'Test data is automatically merged into the training split',
                ],
                'correct_idx': 1,
                'reference': 'An excessively large learning rate causes updates to overshoot the minimum so the loss oscillates or diverges.',
            },
            {
                'question': 'Why does Cross-Entropy Loss penalize a confidently wrong prediction (e.g., predicting p=0.01 for the true class) much more heavily than Mean Squared Error?',
                'choices': [
                    'Because -log(p) grows steeply toward infinity as predicted probability p approaches 0',
                    'Because Cross-Entropy ignores predicted probabilities completely',
                    'Because Mean Squared Error always outputs negative numbers',
                    'Because Cross-Entropy multiplies the loss by zero',
                ],
                'correct_idx': 0,
                'reference': 'Cross-entropy uses negative log likelihood -log(p), which produces strong corrective gradients when the model assigns near-zero probability to the true label.',
            },
            {
                'question': 'How does AdamW improve upon plain Stochastic Gradient Descent when training deep architectures?',
                'choices': [
                    'It adapts per-parameter step sizes using running first and second gradient moments with decoupled weight decay',
                    'It freezes 99% of weights permanently at initialization',
                    'It replaces backpropagation with grid search',
                    'It sets the learning rate to infinity after epoch 1',
                ],
                'correct_idx': 0,
                'reference': 'AdamW tracks exponential moving averages of gradients and squared gradients for adaptive per-parameter updates alongside decoupled weight decay.',
            },
        ],
        'boss_open': [
            {
                'question': 'Explain how the learning rate controls the trade-off between convergence speed and stability in gradient descent.',
                'reference': 'A small learning rate takes tiny stable steps but converges slowly, whereas an overly large learning rate steps too far and can overshoot the minimum or cause the loss to diverge.',
            },
            {
                'question': 'In your own words, how do a loss function and gradient descent work together to train a neural network?',
                'reference': 'The loss function quantifies how far predictions are from the true targets, and gradient descent shifts weights in the negative direction of the loss gradient to iteratively reduce that error.',
            },
        ],
    },

    # =========================================================================
    # CONCEPT 2: Backpropagation & Gradient Stability (Requires C0)
    # =========================================================================
    {
        'concept_id': 2,
        'title': 'Backpropagation & Gradient Stability',
        'prereqs': [0],
        'podcast_hint': (
            '[Podcast Clip 06:05 - 06:20] "Backpropagation applies the calculus chain rule backward '
            'from the output loss to early layers. If local derivatives are huge, gradients explode — '
            'which is why gradient clipping caps the gradient norm!"'
        ),
        'easy_mc': [
            {
                'question': 'Which calculus principle does backpropagation use to compute gradients through multi-layer networks?',
                'choices': [
                    'The chain rule of derivatives applied layer by layer from output to input',
                    'Long division of token IDs without derivatives',
                    'Sorting the vocabulary in reverse order',
                    'Counting the number of files on disk',
                ],
                'correct_idx': 0,
                'reference': 'Backpropagation applies the chain rule of calculus backward from the loss to compute gradients for every layer.',
            },
            {
                'question': 'Why must forward-pass intermediate activations be cached in memory during `model.train()`?',
                'choices': [
                    'Because computing exact local weight gradients during the backward pass depends on those forward activations',
                    'Only to make GPU VRAM look fuller',
                    'To display them as desktop wallpaper',
                    'Because evaluation mode requires backward gradients',
                ],
                'correct_idx': 0,
                'reference': 'During backpropagation, each layer needs its cached forward inputs/activations to compute local derivatives via the chain rule.',
            },
            {
                'question': 'What problem occurs when repeated chain-rule multiplications shrink gradients toward zero in early layers?',
                'choices': [
                    'Vanishing gradients, causing early layers to learn extremely slowly or stall',
                    'Batch size overflow',
                    'Instant 100% test accuracy',
                    'Tokenization mismatch',
                ],
                'correct_idx': 0,
                'reference': 'Vanishing gradients occur when multiplying many small derivatives shrinks gradients near zero so early layers barely update.',
            },
        ],
        'medium_mc': [
            {
                'question': 'How does gradient clipping protect recurrent neural networks (RNNs/GRUs) during backpropagation through time?',
                'choices': [
                    'It deletes half the hidden layers permanently',
                    'It rescales gradients when their norm exceeds a threshold so exploding gradients do not destabilize weights',
                    'It sets all labels in the dataset to zero',
                    'It disables backpropagation and freezes every layer',
                ],
                'correct_idx': 1,
                'reference': 'Gradient clipping rescales gradients whenever their norm exceeds a threshold, preventing exploding gradients from causing unstable weight updates.',
            },
            {
                'question': 'How do residual skip connections (`y = x + F(x)`) help very deep networks avoid vanishing gradients?',
                'choices': [
                    'The identity path `+ x` creates an unimpeded gradient highway with derivative 1 back to earlier layers',
                    'They remove all activation functions from the model',
                    'They restrict networks to at most two layers',
                    'They multiply all gradients by zero',
                ],
                'correct_idx': 0,
                'reference': 'Residual skip connections add the input directly to the block output so gradients flow backward cleanly through the identity path without vanishing.',
            },
            {
                'question': 'How do GRU update and reset gates improve over vanilla RNNs on long student interaction sequences?',
                'choices': [
                    'Gates selectively retain long-term hidden state information and regulate gradient flow across many timesteps',
                    'GRUs process sequences in random order without hidden states',
                    'GRUs require webcam video frames to run',
                    'GRUs only work on sequences of length 1',
                ],
                'correct_idx': 0,
                'reference': 'GRU gating mechanisms control what past memory to keep or update, mitigating vanishing gradients over long sequences.',
            },
        ],
        'boss_open': [
            {
                'question': 'Why do deep or recurrent networks suffer from vanishing or exploding gradients, and how does gradient clipping or gating (LSTM/GRU) help?',
                'reference': 'Repeatedly multiplying derivatives via the chain rule causes gradients to shrink exponentially (vanish) or grow exponentially (explode); gating preserves long-range gradient flow while gradient clipping caps excessive norms.',
            },
            {
                'question': 'Explain in your own words how residual connections (`x + F(x)`) or LayerNorm stabilize gradient flow in deep networks.',
                'reference': 'Residual connections provide a direct identity shortcut so gradients propagate backward without shrinking to zero, while LayerNorm keeps activations well-scaled across layers.',
            },
        ],
    },

    # =========================================================================
    # CONCEPT 3: Overfitting, Dropout & Regularization (Requires C1)
    # =========================================================================
    {
        'concept_id': 3,
        'title': 'Overfitting, Dropout & Regularization',
        'prereqs': [1],
        'podcast_hint': (
            '[Podcast Clip 08:50 - 09:05] "When training loss drops near zero while validation loss '
            'climbs upward, your model is memorizing noise! Dropout fights this by randomly muting '
            'neurons during training so no single pathway can cheat."'
        ),
        'easy_mc': [
            {
                'question': 'What is the classic diagnostic sign that a neural network is overfitting?',
                'choices': [
                    'Training loss keeps decreasing while unseen validation/test loss starts rising',
                    'Both training accuracy and test accuracy are 99.9%',
                    'The GPU fan spins quieter during evaluation',
                    'Batch size equals a power of two',
                ],
                'correct_idx': 0,
                'reference': 'Overfitting appears when training error continues to drop as the model memorizes training data while unseen test error worsens.',
            },
            {
                'question': 'What is the purpose of Early Stopping when training a deep neural network?',
                'choices': [
                    'Halting training and restoring the best checkpoint when validation performance stops improving',
                    'Stopping after the very first mini-batch of Epoch 1',
                    'Evaluating the model directly on training labels only',
                    'Deleting the validation set to train faster',
                ],
                'correct_idx': 0,
                'reference': 'Early stopping monitors held-out validation metrics and restores the peak checkpoint before the model overfits.',
            },
            {
                'question': 'What does L2 weight decay penalize in the optimizer?',
                'choices': [
                    'Overly large weight magnitudes, encouraging simpler smoother models that generalize better',
                    'Small batch sizes',
                    'Fast GPU inference latency',
                    'High validation accuracy',
                ],
                'correct_idx': 0,
                'reference': 'Weight decay shrinks large weights toward zero, constraining model capacity to reduce overfitting.',
            },
        ],
        'medium_mc': [
            {
                'question': 'How is Dropout behavior changed when switching a PyTorch model from `model.train()` to `model.eval()`?',
                'choices': [
                    'All neurons are dropped so the output is always zero',
                    'Random neuron dropping is disabled so the full ensemble of learned weights is active at inference',
                    'Dropout rate increases to 95% during testing',
                    'Weights are randomly re-initialized before every test batch',
                ],
                'correct_idx': 1,
                'reference': 'During evaluation, dropout is turned off so no neurons are randomly zeroed out and predictions are deterministic.',
            },
            {
                'question': 'Why must statistics like item priors or feature scalers be computed strictly on the TRAIN split and never on the TEST split?',
                'choices': [
                    'Computing statistics using test targets leaks future test answers into training features and inflates metrics falsely',
                    'Test splits contain fewer columns than train splits',
                    'Floating-point addition fails on test rows',
                    'Train splits are always sorted alphabetically',
                ],
                'correct_idx': 0,
                'reference': 'Fitting priors or normalizers on the test set causes data leakage, giving the model unfair information from unseen evaluation data.',
            },
            {
                'question': 'How does label smoothing (e.g., `0.03`) help calibrate a neural classifier during training?',
                'choices': [
                    'It prevents logits from growing toward infinity, reducing overconfidence and improving generalization',
                    'It converts all labels into random noise',
                    'It forces every class to have equal 33.3% probability at test time',
                    'It removes the softmax function',
                ],
                'correct_idx': 0,
                'reference': 'Label smoothing softens hard 0/1 targets slightly so the classifier avoids extreme overconfident logits and generalizes better.',
            },
        ],
        'boss_open': [
            {
                'question': 'Explain how Dropout regularization reduces overfitting during neural network training, and how it behaves at test time.',
                'reference': 'Dropout randomly deactivates neurons during training to prevent co-adaptation and force redundant generalizable features, and at test time dropout is disabled so all neurons contribute deterministically.',
            },
            {
                'question': 'How do you recognize overfitting on training curves, and name two techniques to prevent it?',
                'reference': 'Overfitting shows up when training loss keeps falling while validation loss rises; it can be prevented using dropout, weight decay, early stopping, or gathering more diverse training data.',
            },
        ],
    },

    # =========================================================================
    # CONCEPT 4: Attention, Sequence Mastery & Adaptive Control (Requires C2 & C3)
    # =========================================================================
    {
        'concept_id': 4,
        'title': 'Attention, Sequence Mastery & Adaptive Control',
        'prereqs': [2, 3],
        'podcast_hint': (
            '[Podcast Clip 11:30 - 11:45] "Unlike a fixed bottleneck vector, self-attention lets '
            'every token directly attend to every other token regardless of distance — catching '
            'long-range negations and word-order swaps in parallel!"'
        ),
        'easy_mc': [
            {
                'question': 'What key advantage does Transformer self-attention have over sequential RNNs on long text sequences?',
                'choices': [
                    'Every token can directly attend to any other token in O(1) path length without sequential step-by-step bottlenecks',
                    'It ignores word order and treats text as an unordered bag of words',
                    'It requires zero matrix multiplications',
                    'It only reads the first three words of any sentence',
                ],
                'correct_idx': 0,
                'reference': 'Self-attention connects any pair of tokens directly in parallel, capturing long-range dependencies without sequential recurrence bottlenecks.',
            },
            {
                'question': 'In Reinforcement Learning (like our Double-DQN Game Master), what does the Q-value Q(s, a) represent?',
                'choices': [
                    'The expected cumulative discounted future reward of taking action a in state s',
                    'The number of words in question s',
                    'The GPU temperature in Celsius',
                    'The cross-entropy loss on image pixels',
                ],
                'correct_idx': 0,
                'reference': 'Q(s, a) estimates the expected total discounted future return starting from state s, taking action a, and acting optimally thereafter.',
            },
            {
                'question': 'Why do Transformers require positional encodings or relative position attention (like DeBERTa-v3)?',
                'choices': [
                    'Because raw dot-product self-attention is permutation-invariant and would otherwise ignore word order',
                    'To sort every sentence alphabetically before grading',
                    'To compress all tokens into a single integer',
                    'To disable GPU parallelism',
                ],
                'correct_idx': 0,
                'reference': 'Without positional information, self-attention treats tokens as an unordered set and could not distinguish inverted word order.',
            },
        ],
        'medium_mc': [
            {
                'question': 'Why does a Cross-Encoder (jointly encoding Reference + Student Answer) outperform simple keyword overlap when grading short answers?',
                'choices': [
                    'It counts shared letters instead of words',
                    'Cross-attention compares words in context across both sentences, catching negations, word swaps, and paraphrases',
                    'It grades purely based on sentence character length',
                    'It ignores the reference answer completely',
                ],
                'correct_idx': 1,
                'reference': 'A Cross-Encoder applies bidirectional cross-attention across both texts simultaneously, recognizing true paraphrases while rejecting negations and word-order inversions.',
            },
            {
                'question': 'How does Double DQN fix the systematic Q-value overestimation bias of standard DQN?',
                'choices': [
                    'It uses the online network to select the best next action and the slowly-updated target network to evaluate its Q-value',
                    'It multiplies all rewards by zero',
                    'It deletes the replay buffer after every episode',
                    'It picks actions purely at random during evaluation',
                ],
                'correct_idx': 0,
                'reference': 'Double DQN decouples action selection (online network) from action evaluation (target network) to prevent maximization bias.',
            },
            {
                'question': 'In scaled dot-product attention `softmax(Q K^T / sqrt(d_k)) V`, why do we divide by `sqrt(d_k)`?',
                'choices': [
                    'To keep dot products from growing too large in high dimensions, which would push softmax into saturated near-zero-gradient regions',
                    'To convert probabilities into negative numbers',
                    'To delete the Value matrix V',
                    'To double the sequence length',
                ],
                'correct_idx': 0,
                'reference': 'Scaling by sqrt(d_k) prevents large dot products from saturating the softmax function and causing vanishing attention gradients.',
            },
        ],
        'boss_open': [
            {
                'question': 'Explain why naive keyword matching fails at grading student explanations and how a Transformer Cross-Encoder solves it.',
                'reference': 'Keyword matching is fooled by negations and inverted word order that share the same words, and unfairly penalizes valid paraphrases, whereas a Transformer Cross-Encoder attends to contextual meaning and word relationships across both answers.',
            },
            {
                'question': 'How do Deep Knowledge Tracing (DKT), an RL Game Master, and a Cross-Encoder Answer Grader work together in an adaptive tutoring game?',
                'reference': 'The Cross-Encoder grades open-text student answers semantically, DKT updates concept mastery probabilities from correctness and behavioral telemetry, and the RL Game Master selects the optimal next scaffold, review, or boss challenge to maximize learning flow.',
            },
        ],
    },
]


def load_course_package(course_id: str = 'vision_yolo_ocr') -> dict:
    """
    Loads a structured 5-concept course package by ID:
      - 'vision_yolo_ocr': Deep Learning Computer Vision (CNNs/FPN, IoU/NMS, YOLO, OCR/CRNN/CTC, Pipelines)
      - 'dl_foundations':  Deep Learning & Adaptive AI Foundations (DEFAULT_CURRICULUM)
    """
    if course_id == 'vision_yolo_ocr' and os.path.exists(VISION_CURRICULUM_JSON):
        with open(VISION_CURRICULUM_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {
        'course_id': 'dl_foundations',
        'title': 'Deep Learning & Neural Foundations',
        'subtitle': 'Activations, Gradient Descent, Backprop, Regularization & Attention',
        'mentor': 'Prof. Ada (Willowbrook Research Lab)',
        'concepts': DEFAULT_CURRICULUM,
    }


def list_available_courses() -> list:
    courses = []
    if os.path.exists(VISION_CURRICULUM_JSON):
        courses.append({
            'course_id': 'vision_yolo_ocr',
            'title': 'Deep Learning Vision: CNNs, YOLO & OCR',
            'subtitle': 'Object Detection, Feature Pyramids, CRNN/CTC & Vision Systems',
        })
    courses.append({
        'course_id': 'dl_foundations',
        'title': 'Deep Learning & Neural Foundations',
        'subtitle': 'Activations, Gradient Descent, Backprop, Regularization & Attention',
    })
    return courses

