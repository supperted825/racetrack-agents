## ME5406 Project 2


A reinforcement learning project based on https://github.com/eleurent/highway-env.

<br/>

## How to Use  

First install dependencies.

```
pip install -r requirements.txt
```

Then, clone the repository.

```
git clone https://github.com/supperted825/ME5406P2.git
```

To run experiments, run main.py while specifying options. See main.py for available options.

```
python main.py [--opts]
```

<br/>

## Developing

Create a branch with the name of the feature that you are working on in ~/highway_env.

```
git co -b <branch name>
```

Now, you can edit the files as normal and develop.

To register changes, be sure to stage the files by running the following at the root of the repo folder.

```
git add .
```

Throughout, you can commit changes with the following command. This doesn't push your changes online yet.

```
git commit -m "your message"
```

To submit your changes to the main repo, you will have to push with:

```
git push
```

Now, if you head back to the repo online, you should see "Compare and Pull Request". Click on it and enter a short description of your changes. Then you can select a reviewer and submit the code for merging with a pull request.