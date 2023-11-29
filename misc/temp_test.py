def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. \
                    For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    logits = outputs.get("logits")
    labels = inputs.get("labels")

    # Apply softmax to logits to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the predicted tokens (highest probability)
    _, predictions = torch.max(probs, dim=-1)

    wrong = (torch.sum(predictions != labels, dim=-1).float()) / labels.shape[-1]

    # Mean loss over the batch
    wrong = wrong.mean()

    loss *= math.pow(wrong, 2)

    return (loss, outputs) if return_outputs else loss
