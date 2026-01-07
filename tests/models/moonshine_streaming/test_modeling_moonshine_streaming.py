# coding=utf-8
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch MoonshineStreaming model."""

import copy
import unittest

from transformers import MoonshineStreamingConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        MoonshineStreamingForConditionalGeneration,
        MoonshineStreamingModel,
    )

from datasets import load_dataset


class MoonshineStreamingModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=1000,
        is_training=False,
        use_labels=False,
        vocab_size=147,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        ffn_mult=4,
        use_swiglu_encoder=False,
        use_swiglu_decoder=True,
        decoder_rotary_dim=4,
        decoder_start_token_id=85,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.ffn_mult = ffn_mult
        self.use_swiglu_encoder = use_swiglu_encoder
        self.use_swiglu_decoder = use_swiglu_decoder
        self.decoder_rotary_dim = decoder_rotary_dim
        self.decoder_start_token_id = decoder_start_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_ids = torch.tensor(self.batch_size * [[self.decoder_start_token_id]], device=torch_device)
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)

        config = self.get_config()

        return config, input_values, attention_mask, decoder_input_ids, decoder_attention_mask

    def get_config(self):
        return MoonshineStreamingConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            encoder_num_hidden_layers=self.num_hidden_layers,
            decoder_num_hidden_layers=self.num_hidden_layers,
            encoder_num_attention_heads=self.num_attention_heads,
            decoder_num_attention_heads=self.num_attention_heads,
            ffn_mult=self.ffn_mult,
            use_swiglu_encoder=self.use_swiglu_encoder,
            use_swiglu_decoder=self.use_swiglu_decoder,
            decoder_rotary_dim=self.decoder_rotary_dim,
            decoder_start_token_id=self.decoder_start_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def check_output_attentions(self, config, input_values, attention_mask):
        model = MoonshineStreamingModel(config=config)
        model.to(torch_device)
        model.train()

        outputs = model(input_values, attention_mask=attention_mask, output_attentions=True)
        self.parent.assertTrue(len(outputs.attentions) > 0)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask, decoder_input_ids, decoder_attention_mask = (
            self.prepare_config_and_inputs()
        )
        inputs_dict = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_torch
class MoonshineStreamingModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (MoonshineStreamingModel, MoonshineStreamingForConditionalGeneration) if is_torch_available() else ()
    )
    # Doesn't run generation tests. TODO (eustache): remove this line and then make CI green
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "automatic-speech-recognition": MoonshineStreamingForConditionalGeneration,
            "feature-extraction": MoonshineStreamingModel,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MoonshineStreamingModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MoonshineStreamingConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_can_init_all_missing_weights(self):
        self.skipTest("MoonshineStreaming uses special parameter initialization that conflicts with this test")

    def test_init_weights_can_init_buffers(self):
        self.skipTest("MoonshineStreaming uses special buffer initialization that conflicts with this test")

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", 1)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()

            subsampled_encoder_seq_length = model._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model._get_feat_extract_output_lengths(encoder_key_length)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_hidden_states_output
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)

                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            decoder_input_ids = inputs.pop("decoder_input_ids", None)
            inputs.pop("decoder_attention_mask", None)

            wte = model.get_input_embeddings()
            inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_resize_tokens_embeddings
    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is False")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # make sure that decoder_input_ids are resized
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    # Copied from tests.models.whisper.test_modeling_whisper.WhisperModelTest.test_resize_embeddings_untied
    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is False")

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untie embeddings")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)
            model.eval()

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
class MoonshineStreamingModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.processor_tiny = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        self.processor_small = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-small")
        self.processor_medium = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-medium")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_tiny_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -14.032719612121582, 0.24530434608459473, 3.2066287994384766, -13.895689010620117, -13.833709716796875,
            -13.876998901367188, -13.8271484375, -13.82671070098877, -13.913274765014648, -13.837194442749023,
            -13.949264526367188, -13.937505722045898, -13.791865348815918, -13.94484806060791, -13.892528533935547,
            -13.906136512756348, -13.92868423461914, -13.768369674682617, -13.811952590942383, -13.758645057678223,
            -13.85223388671875, -13.922545433044434, -13.863603591918945, -13.820209503173828, -13.886418342590332,
            -13.854723930358887, -13.924177169799805, -13.828561782836914, -13.79515266418457,
            -13.905285835266113,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_small_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        inputs = self.processor_small(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -9.391862869262695, -1.4379109144210815, 2.581448554992676, -9.705957412719727, -9.678482055664062,
            -9.695582389831543, -9.709806442260742, -9.711294174194336, -9.687347412109375, -9.704854011535645,
            -9.696181297302246, -9.717063903808594, -9.746146202087402, -9.697282791137695, -9.700727462768555,
            -9.69192123413086, -9.64999008178711, -9.714788436889648, -9.700082778930664, -9.664180755615234,
            -9.701624870300293, -9.709983825683594, -9.694823265075684, -9.670870780944824, -9.679281234741211,
            -9.67048454284668, -9.67465877532959, -9.718086242675781, -9.720056533813477, -9.705082893371582,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)

    @slow
    def test_medium_logits_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        inputs = self.processor_medium(self._load_datasamples(1), return_tensors="pt")
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            -9.467172622680664, -1.8976192474365234, 1.2782490253448486, -10.107687950134277, -10.172184944152832,
            -10.12191390991211, -10.123106956481934, -10.169483184814453, -10.22452163696289, -10.059475898742676,
            -10.113517761230469, -10.192845344543457, -10.161031723022461, -10.134140014648438, -10.17141056060791,
            -10.206125259399414, -10.160161972045898, -10.18079662322998, -10.109098434448242, -10.158792495727539,
            -10.081502914428711, -10.083662033081055, -10.123435974121094, -10.159687042236328, -10.088380813598633,
            -10.12335205078125, -10.163549423217773, -10.109436988830566, -10.104336738586426, -10.056718826293945,
        ])
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][0, :30].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)

    @slow
    def test_tiny_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        inputs = self.processor_tiny(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-12.678281784057617, 0.18974974751472473, 2.8350114822387695, -12.467742919921875, -12.394091606140137, -12.435752868652344, -12.408197402954102, -12.414243698120117, -12.502815246582031, -12.418495178222656],
                [-13.185888290405273, -3.4433515071868896, 4.022008419036865, -12.958669662475586, -13.035518646240234, -13.005077362060547, -13.053488731384277, -12.953858375549316, -13.049406051635742, -12.956440925598145],
                [-10.096810340881348, -4.12624454498291, 4.332856178283691, -10.00971794128418, -10.03563117980957, -9.97742748260498, -9.972275733947754, -10.001754760742188, -10.023305892944336, -10.070893287658691],
                [-11.445636749267578, -3.0520095825195312, 3.810762643814087, -11.261188507080078, -11.298276901245117, -11.321820259094238, -11.292222023010254, -11.325114250183105, -11.356322288513184, -11.292496681213379],
            ],
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_small_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        inputs = self.processor_small(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-9.669126510620117, -1.3994865417480469, 2.946110248565674, -9.916278839111328, -9.896139144897461, -9.891678810119629, -9.903473854064941, -9.913434982299805, -9.892568588256836, -9.902320861816406],
                [-9.754722595214844, 0.2890121042728424, 3.0636770725250244, -9.893263816833496, -9.948053359985352, -9.970532417297363, -9.920491218566895, -9.890860557556152, -9.965429306030273, -9.952640533447266],
                [-10.451912879943848, -0.35098984837532043, 3.2526659965515137, -10.167275428771973, -10.189103126525879, -10.2238130569458, -10.179900169372559, -10.235410690307617, -10.16427230834961, -10.219122886657715],
                [-10.0325288772583, -1.4704744815826416, 3.414222240447998, -9.494173049926758, -9.504581451416016, -9.561887741088867, -9.513245582580566, -9.518109321594238, -9.49057388305664, -9.498723030090332],
            ]
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_medium_logits_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        inputs = self.processor_medium(self._load_datasamples(4), return_tensors="pt", padding=True)
        inputs.to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [-9.47853946685791, -1.6993076801300049, 1.2424887418746948, -10.110610961914062, -10.16978931427002, -10.129164695739746, -10.13774585723877, -10.168231010437012, -10.228961944580078, -10.08574104309082],
                [-10.049469947814941, -2.302199602127075, 2.3719050884246826, -10.332446098327637, -10.39849853515625, -10.327662467956543, -10.365711212158203, -10.438501358032227, -10.46987247467041, -10.402321815490723],
                [-9.312597274780273, -0.6024600863456726, 1.8864504098892212, -9.744148254394531, -9.794787406921387, -9.759222984313965, -9.746228218078613, -9.768878936767578, -9.794204711914062, -9.792024612426758],
                [-8.998611450195312, -0.6439317464828491, 1.2607133388519287, -8.98252010345459, -9.070300102233887, -8.981226921081543, -8.99964427947998, -9.034867286682129, -9.038430213928223, -9.001120567321777],
            ]
        )
        # fmt: on
        torch.testing.assert_close(outputs.logits[0][:, :10].cpu(), EXPECTED_LOGITS, rtol=2e-4, atol=2e-4)

    @slow
    def test_tiny_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_tiny(audio_array, return_tensors="pt")
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_small_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_small(audio_array, return_tensors="pt")
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_small.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_medium_generation_single(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        audio_array = self._load_datasamples(1)
        inputs = self.processor_medium(audio_array, return_tensors="pt")
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_medium.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_tiny(audio_array, return_tensors="pt", padding=True)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_tiny.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mr. Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and a roast be",
            "He has grieved doubts whether Sir Frederick Layton's work is really Greek after all",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_small_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-small")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_small(audio_array, return_tensors="pt", padding=True)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_small.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mister Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mister Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and roast beef",
            "He has grave doubts whether Sir Frederick Layton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_medium_generation_batch(self):
        model = MoonshineStreamingForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-streaming-medium")
        model.to(torch_device)

        audio_array = self._load_datasamples(4)
        inputs = self.processor_medium(audio_array, return_tensors="pt", padding=True)
        inputs.to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        transcript = self.processor_medium.batch_decode(generated_ids, skip_special_tokens=True)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            "Mister Quilter is the apostle of the middle classes, and we are glad to welcome",
            "Nor is Mister Quilter's manner less interesting than his matter.",
            "He tells us that at this festive season of the year, with Christmas and roast beef",
            "He has grave doubts whether Sir Frederick Leighton's work is really Greek after all,",
        ]
        # fmt: on

        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)
