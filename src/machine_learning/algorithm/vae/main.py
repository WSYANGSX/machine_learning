from VAE import VAE


def main():
    vae = VAE(
        z_dim=128,
        config_file="./Machine learning/VAE/config.yaml",
        device="cuda",
    )

    vae.train_model()

    vae.visualize_reconstruction()


if __name__ == "__main__":
    main()
