from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Text
import logging
import requests

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT, INTENT_RANKING_KEY, TEXT_TOKENS
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.nlu.classifiers.classifier import IntentClassifier
import rasa.utils.io
import rasa.shared.utils.io
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    MEMOIZATION_POLICY_PRIORITY,
    DEFAULT_MAX_HISTORY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)

logger = logging.getLogger(__name__)


try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print('No dotenv')


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class NLUModify(GraphComponent, EntityExtractorMixin):
    """Maps entities to their synonyms if they appear in the training data."""

    def __init__(
        self,
        config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates the mapper.

        Args:
            config: The mapper's config.
            model_storage: Storage which the component can use to persist and load
                itself.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            synonyms: A dictionary of previously known synonyms.
        """
        self._config = config
        self._model_storage = model_storage
        self._resource = resource

        self.synonyms = synonyms if synonyms else {}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> NLUModify:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, synonyms)

    def train(self, training_data: TrainingData) -> Resource:
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        """Modifies entities attached to message to resolve synonyms.

        Args:
            messages: List containing the latest user message

        Returns:
            List containing the latest user message with entities resolved to
            synonyms if there is a match.
        """
        try:
            url = self._config['url']
            body = {
                'messages': [],
                'config': self._config
            }
            for msg in messages:
                body['messages'].append({
                    TEXT: msg.get(TEXT),
                    ENTITIES: msg.get(ENTITIES, []),
                    INTENT: msg.get(INTENT, {}),
                })

            resp = requests.post(url, json=body)
            data = resp.json()
            if data['status'] != 'SUCCESS':
                return messages
            else:
                resMsgs = data['data']
                index = 0   
                for message in messages:
                    resMsg = resMsgs[index]
                    for key in resMsg:
                        message.set(key, resMsg[key], add_to_output=True)
                    index += 1
                return messages
        except Exception as e:
            print('LogException', e)
            return messages

    # Adapt to get path from model storage and resource
    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> NLUModify:
        """Loads trained component (see parent class for full docstring)."""
        synonyms = None

        return cls(config, model_storage, resource, synonyms)




@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ],
    is_trainable=False,
)
class HybridDIETClassifier(GraphComponent, IntentClassifier, EntityExtractorMixin):
    """Maps entities to their synonyms if they appear in the training data."""

    def __init__(
        self,
        config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates the mapper.

        Args:
            config: The mapper's config.
            model_storage: Storage which the component can use to persist and load
                itself.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            synonyms: A dictionary of previously known synonyms.
        """
        self._config = config
        self._model_storage = model_storage
        self._resource = resource

        self.synonyms = synonyms if synonyms else {}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> HybridDIETClassifier:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, synonyms)

    def train(self, training_data: TrainingData) -> Resource:
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        """Modifies entities attached to message to resolve synonyms.

        Args:
            messages: List containing the latest user message

        Returns:
            List containing the latest user message with entities resolved to
            synonyms if there is a match.
        """
        try:
            url = self._config['url']
            body = {
                'messages': [],
                'config': self._config
            }
            for msg in messages:
                body['messages'].append({
                    TEXT: msg.get(TEXT),
                    TEXT_TOKENS: msg.get(TEXT_TOKENS),
                    ENTITIES: msg.get(ENTITIES, []),
                    INTENT: msg.get(INTENT, {}),
                    INTENT_RANKING_KEY: msg.get(INTENT_RANKING_KEY, []),
                })

            resp = requests.post(url, json=body)
            data = resp.json()
            if data['status'] != 'SUCCESS':
                return messages
            else:
                resMsgs = data['data']
                index = 0
                for message in messages:
                    resMsg = resMsgs[index]
                    for key in resMsg:
                        message.set(key, resMsg[key], add_to_output=True)
                    index += 1
                return messages
        except Exception as e:
            logger.error('LogException', e)
            return messages

    # Adapt to get path from model storage and resource
    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> HybridDIETClassifier:
        """Loads trained component (see parent class for full docstring)."""
        synonyms = None

        return cls(config, model_storage, resource, synonyms)




@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=False
)
class HybridPolicy(Policy):
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            POLICY_PRIORITY: 1
        }

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        # result = [0.1] * domain.num_actions
        try:
            url = self.config['url']
            trackerObj = tracker.current_state()
            trackerObj['events'] = [e.as_dict() for e in tracker.events]
            domainObj = domain.as_dict()
            domainObj['allActions'] = domain.action_names_or_texts
            body = {
                'tracker': trackerObj,
                'domain': domainObj,
                'config': self.config,
            }
            resp = requests.post(url, json=body)
            data = resp.json()

            if data['status'] != 'SUCCESS':
                return self._prediction(self._default_predictions(domain))
            else:
                resMsgs = data['data']
                return self._prediction(resMsgs['predictions'])
        except Exception as e:
            print('LogException', e)
            return self._prediction(self._default_predictions(domain))

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> HybridPolicy:
        """Loads a trained policy (see parent class for full docstring)."""
        featurizer = None

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            featurizer=featurizer
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        return self._resource
