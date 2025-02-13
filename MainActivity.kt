package com.example.thesistest1

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.thesistest1.ui.theme.Thesistest1Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val ppl = listOf("hi", "hello", "john")

        setContent {
            Thesistest1Theme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(name = "henson", modifier = Modifier.padding(innerPadding))
                    GreetingPreview()


                    LazyColumn {
                        items(ppl){
                            ListItem(it)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Surface(color = Color.Blue) {
        Text(
            text = "Hello $name!",
            modifier = modifier.padding(24.dp)
        )
    }


}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    Thesistest1Theme {
        //Greeting("michelle")
        Greeting(",michelle gpt is gay")


    }
}



@Composable
fun ListItem(name : String){
    Card(modifier = Modifier.fillMaxSize()
        .padding(24.dp)){
        Image(painter = painterResource(id = R.drawable.baseline_person_24), contentDescription = "photo")
        Text(text = name, modifier = Modifier.padding(12.dp))

    }


}