## ME5406 Project 2


A project based on https://github.com/eleurent/highway-env.

<br/>

## How to Use  

First install the highway_env package in your virtual environment.

```
pip install highway_env==1.4
```

Navigate to the installation directory of highway_env (~/site-packages/highway_env) and remove everything.

Next, clone this repo.

```
git clone https://github.com/supperted825/ME5406P2.git
```

Once done, move all the files as is out of the ME5406P2 folder, so that the folder root is ~/highway_env.

<br/>

## How to Develop

Following from the previous steps, create a branch with the name of the feature that you are working on in ~/highway_env.

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

# ME5406P2
