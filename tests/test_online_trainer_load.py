import pickle

from tradingbot_ibkr.models.online_trainer import OnlineTrainer


def test_online_trainer_load_ignores_incompatible_pickle(tmp_path):
    # Arrange: create an incompatible pickle (a plain dict) at a temp path
    trainer = OnlineTrainer()
    trainer.path = tmp_path / "online_model.pkl"
    with open(trainer.path, "wb") as f:
        pickle.dump({"not": "a model"}, f)

    # Act: load the incompatible file
    trainer.load()

    # Assert: trainer.model was reset to a compatible default model
    assert hasattr(trainer.model, "predict_proba_one")
    assert hasattr(trainer.model, "learn_one")

    # And the APIs work without raising
    p = trainer.predict_proba({"x": 0.0})
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

    # learn_one should also execute without error
    trainer.learn_one({"x": 1.0}, 1)
